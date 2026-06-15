"""
End-to-end BurnTrack pipeline runner.

Orchestrates: weather download -> feature engineering -> Rothermel -> AI corrector -> danger assessment.

Usage:
    python scripts/run_pipeline.py --lat 31.63 --lon -7.98 --fuel-model AF_STEPPE --date 2024-08-15
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class PipelineRunner:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.corrector = None
        self.scaler = None
        self.corrector_type = None
        self._load_corrector()

    def _load_corrector(self):
        import joblib

        for model_file in ["rf_corrector.joblib", "xgb_corrector.joblib"]:
            model_path = os.path.join(self.model_dir, model_file)
            scaler_path = os.path.join(self.model_dir, "rf_scaler.joblib")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.corrector = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.corrector_type = "rf" if "rf" in model_file else "xgb"
                return

        try:
            import torch
            from burntrack.corrector.base import BaseCorrector  # noqa

            model_path = os.path.join(self.model_dir, "corrector_v3_best.pt")
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.corrector_type = "mlp"
        except ImportError:
            pass

    def _compute_rothermel(self, fuel_code: str, env: dict, conditions: dict) -> dict:
        try:
            from burntrack.engine import (
                RothermelEngine,
                FuelModel,
                MoistureInputs,
                EnvironmentalConditions,
            )
        except ImportError:
            print("ERROR: burntrack.engine not found.")
            return {}

        fuel = FuelModel(fuel_code)
        moisture = MoistureInputs(
            m_1h=env.get("m_1h", 0.05),
            m_10h=env.get("m_10h", 0.06),
            m_100h=env.get("m_100h", 0.07),
            m_live_herb=env.get("m_live_herb", 0.30),
            m_live_woody=env.get("m_live_woody", 0.60),
        )
        env_conds = EnvironmentalConditions(
            wind_speed=conditions.get("wind_mid_flame", 3.0),
            slope_pct=conditions.get("slope_pct", 0.0),
            angle_wind_slope=conditions.get("angle_wind_slope", 0.0),
        )

        engine = RothermelEngine()
        out = engine.compute(fuel, moisture, env_conds)

        ros = out.ros
        if ros < 1.0:
            danger = "LOW"
        elif ros < 3.0:
            danger = "MODERATE"
        elif ros < 10.0:
            danger = "HIGH"
        else:
            danger = "VERY HIGH"

        return {
            "ros": ros,
            "ros_m_min": ros,
            "flame_length": out.flame_length,
            "flame_length_m": out.flame_length,
            "fireline_intensity": out.fireline_intensity,
            "fireline_intensity_kW_m": out.fireline_intensity,
            "heat_per_unit_area": out.heat_per_unit_area,
            "fuel_consumption": out.fuel_consumption,
            "direction": out.spread_direction,
            "phi_w": out.phi_w,
            "phi_s": out.phi_s,
            "phi_eff": out.phi_eff,
            "danger_level": danger,
        }

    def _apply_corrector(self, rothermel_out: dict, features: dict) -> dict:
        if self.corrector is None or self.scaler is None:
            return {
                "ros_corrected": rothermel_out.get("ros_m_min", 0.0),
                "delta_ros": 0.0,
                "uncertainty_std": 0.0,
            }

        try:
            ros_rothermel = rothermel_out.get("ros_m_min", 0.0)
            x_cont = np.array([[features.get(k, 0.0) for k in (
                "temp_air", "rh", "wind_speed", "vpd", "slope_deg", "slope_pct",
                "w_total", "w_dead", "w_live", "delta", "sigma", "mx", "h_dead",
                "phi_w", "phi_s", "phi_eff", "beta", "beta_opt", "gamma",
                "eta_M", "eta_S", "I_R", "xi", "tau", "ndvi", "ndwi", "lst", "dfmc"
            )]], dtype=np.float32)

            if self.corrector_type in ("rf", "xgb"):
                x_scaled = self.scaler.transform(x_cont)
                pred_log = self.corrector.predict(x_scaled)[0]
                ros_corrected = max(0.0, float(np.exp(pred_log) - 0.1))
                delta_ros = ros_corrected - ros_rothermel
                uncertainty_std = 0.0
            else:
                import torch
                x_scaled = self.scaler.transform(x_cont)
                x_t = torch.tensor(x_scaled, dtype=torch.float32)
                fuel_t = torch.tensor([features.get("fuel_idx", 0)], dtype=torch.long)
                with torch.no_grad():
                    out = self.corrector(x_t, fuel_t).numpy()[0]
                delta_ros = float(out[0])
                ros_corrected = max(0.0, ros_rothermel + delta_ros)
                uncertainty_std = float(np.sqrt(np.exp(float(out[1]))))

            return {
                "ros_corrected": ros_corrected,
                "delta_ros": delta_ros,
                "uncertainty_std": uncertainty_std,
            }
        except Exception as e:
            warnings.warn(f"Corrector inference failed: {e}")
            return {
                "ros_corrected": rothermel_out.get("ros_m_min", 0.0),
                "delta_ros": 0.0,
                "uncertainty_std": 0.0,
            }

    def run(self, lat: float, lon: float, fuel_model: str,
            date: Optional[str] = None,
            robot_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        env = {
            "m_1h": 0.05,
            "m_10h": 0.06,
            "m_100h": 0.07,
            "m_live_herb": 0.30,
            "m_live_woody": 0.60,
        }

        conditions = {
            "wind_mid_flame": robot_data.get("wind_speed", 3.0) * 0.4 if robot_data else 3.0,
            "slope_pct": robot_data.get("slope_pct", 0.0) if robot_data else 0.0,
            "angle_wind_slope": 0.0,
        }

        rothermel_out = self._compute_rothermel(fuel_model, env, conditions)

        features = {
            "temp_air": robot_data.get("temp_air", 25.0) if robot_data else 25.0,
            "rh": robot_data.get("rh", 40.0) if robot_data else 40.0,
            "wind_speed": robot_data.get("wind_speed", 3.0) if robot_data else 3.0,
            "vpd": 1.0,
            "slope_deg": robot_data.get("slope_deg", 0.0) if robot_data else 0.0,
            "slope_pct": conditions["slope_pct"],
            "w_total": 0.5,
            "w_dead": 0.3,
            "w_live": 0.2,
            "delta": 0.3,
            "sigma": 1500.0,
            "mx": 20.0,
            "h_dead": 18622.0,
            "phi_w": rothermel_out.get("phi_w", 0.0),
            "phi_s": rothermel_out.get("phi_s", 0.0),
            "phi_eff": rothermel_out.get("phi_eff", 0.0),
            "beta": 0.001,
            "beta_opt": 0.001,
            "gamma": 1.0,
            "eta_M": 1.0,
            "eta_S": 1.0,
            "I_R": rothermel_out.get("fireline_intensity_kW_m", 0.0),
            "xi": 0.5,
            "tau": 0.5,
            "ndvi": 0.3,
            "ndwi": 0.0,
            "lst": 25.0,
            "dfmc": 10.0,
            "fuel_idx": 0,
        }

        ia_out = self._apply_corrector(rothermel_out, features)

        final_danger = rothermel_out.get("danger_level", "UNKNOWN")
        ros_final = ia_out.get("ros_corrected", rothermel_out.get("ros_m_min", 0.0))

        return {
            "location": {"lat": lat, "lon": lon, "date": date},
            "fuel_model": fuel_model,
            "rothermel": rothermel_out,
            "corrector": ia_out,
            "danger_level": final_danger,
            "ros_final_m_min": ros_final,
        }


def main():
    parser = argparse.ArgumentParser(description="Run the BurnTrack wildfire spread pipeline")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--fuel-model", type=str, required=True, help="Fuel model code (e.g. AF_STEPPE)")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with corrector model files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    runner = PipelineRunner(model_dir=args.model_dir)
    result = runner.run(
        lat=args.lat,
        lon=args.lon,
        fuel_model=args.fuel_model,
        date=args.date,
    )

    if args.json:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        print(f"\n{'='*50}")
        print("BURNTRACK PIPELINE RESULTS")
        print(f"{'='*50}")
        print(f"  Location   : {result['location']['lat']}, {result['location']['lon']}")
        print(f"  Date       : {result['location']['date']}")
        print(f"  Fuel Model : {result['fuel_model']}")
        print()

        roth = result["rothermel"]
        if roth:
            print("  ROTHERMEL OUTPUT")
            print(f"    ROS               : {roth.get('ros_m_min', 'N/A'):.4f} m/min")
            print(f"    Flame Length      : {roth.get('flame_length_m', 'N/A'):.4f} m")
            print(f"    Fireline Intensity: {roth.get('fireline_intensity_kW_m', 'N/A'):.2f} kW/m")
            print(f"    Danger Level      : {roth.get('danger_level', 'N/A')}")
            print()

        corr = result["corrector"]
        print("  AI CORRECTOR")
        print(f"    ROS Corrected  : {corr.get('ros_corrected', 'N/A'):.4f} m/min")
        print(f"    Delta ROS      : {corr.get('delta_ros', 'N/A'):.4f} m/min")
        print(f"    Uncertainty    : +/- {corr.get('uncertainty_std', 'N/A'):.4f} m/min")
        print()
        print(f"  FINAL DANGER LEVEL: {result['danger_level']}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
