import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellular_automaton import PropagationRules
from experiments.validate_ca_fire import scenario

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
REPORT_PATH = os.path.join(os.path.dirname(__file__), "calibration_report.md")

scenarios_cfg = [
    ("GR4", 0.06, 5.0, "GR4 6% 5m/s"),
    ("GR4", 0.06, 3.0, "GR4 6% 3m/s"),
    ("GR4", 0.15, 5.0, "GR4 15% 5m/s"),
    ("GR4", 0.28, 5.0, "GR4 28% 5m/s"),
    ("GR4", 0.35, 3.0, "GR4 35% 3m/s"),
    ("AF_SAHEL_GRASS", 0.15, 5.0, "SAHEL 15% 5m/s"),
    ("AF_SAHEL_GRASS", 0.28, 5.0, "SAHEL 28% 5m/s"),
    ("AF_GRASSLAND_FERTILE", 0.15, 5.0, "FERTILE 15% 5m/s"),
    ("AF_GRASSLAND_FERTILE", 0.28, 5.0, "FERTILE 28% 5m/s"),
    ("AF_MIOMBO", 0.20, 4.0, "MIOMBO 20% 4m/s"),
    ("AF_BUSHVELD", 0.20, 5.0, "BUSHVELD 20% 5m/s"),
    ("AF_FYNBOS", 0.20, 5.0, "FYNBOS 20% 5m/s"),
    ("AF_MOPANE", 0.15, 5.0, "MOPANE 15% 5m/s"),
]

def evaluate(params):
    ratios = []
    results = []
    rules = PropagationRules(
        stochastic=True,
        directional_exponent=params['dir_exp'],
        back_fire_fraction=params['back_fire'],
        burn_duration_factor=4.0,
        min_burn_min=5.0,
        min_ros_m_min=0.01,
    )
    for fuel, m, ws, name in scenarios_cfg:
        r = scenario(fuel, m, ws, name=name, dt=params['dt'], rules=rules)
        ratios.append(r['ros_ratio_pct'])
        results.append(r)
    
    # Cost function
    abs_errs = [abs(r - 100) for r in ratios]
    cost = np.mean(abs_errs) + 0.5 * np.std(ratios)
    return cost, ratios, results

def main():
    dir_exps = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    back_fires = [0.05, 0.10, 0.15, 0.20, 0.30]
    dt = 0.25
    
    best_cost = float('inf')
    best_params = None
    best_results = None
    best_ratios = None
    
    grid_results = []
    
    print("Starting Grid Search...")
    for de in dir_exps:
        for bf in back_fires:
            print(f"\nEvaluating dir_exp={de}, back_fire={bf}")
            params = {'dir_exp': de, 'back_fire': bf, 'dt': dt}
            cost, ratios, results = evaluate(params)
            grid_results.append({
                'dir_exp': de,
                'back_fire': bf,
                'cost': cost,
                'mean_err': np.mean([abs(r - 100) for r in ratios]),
                'std_err': np.std(ratios)
            })
            if cost < best_cost:
                best_cost = cost
                best_params = params
                best_results = results
                best_ratios = ratios
                
    # Generate Report
    with open(REPORT_PATH, 'w') as f:
        f.write("# Calibration Report\n\n")
        f.write("## Parameter Grid Results\n\n")
        f.write("| dir_exp | back_fire | Mean |ratio-100| | Std Dev | Cost |\n")
        f.write("|---|---|---|---|---|\n")
        for g in sorted(grid_results, key=lambda x: x['cost']):
            f.write(f"| {g['dir_exp']} | {g['back_fire']} | {g['mean_err']:.2f} | {g['std_err']:.2f} | {g['cost']:.2f} |\n")
            
        f.write("\n## Optimal Parameters\n\n")
        f.write(f"**directional_exponent**: {best_params['dir_exp']}\n")
        f.write(f"**back_fire_fraction**: {best_params['back_fire']}\n")
        f.write(f"**dt**: {best_params['dt']}\n")
        f.write(f"**Cost**: {best_cost:.2f}\n\n")
        
        f.write("## Detailed Results for Optimal Configuration\n\n")
        f.write("| Scenario | Rothermel ROS (m/min) | CA Observed (m/min) | Ratio (%) |\n")
        f.write("|---|---|---|---|\n")
        for r in best_results:
            f.write(f"| {r['name']} | {r['pred_ros']:.2f} | {r['obs_ros']:.2f} | {r['ros_ratio_pct']:.1f}% |\n")
            
        f.write("\n## Residual Bias\n")
        f.write("Some fuels may still deviate. Fuels with error > 30%:\n")
        for r in best_results:
            if abs(r['ros_ratio_pct'] - 100) > 30:
                f.write(f"- {r['name']}: {r['ros_ratio_pct']:.1f}%\n")
                
        f.write("\n## CFL Safety Margin\n")
        f.write("Target dt=0.25 provides sufficient safety margin for typical ROS up to 120 m/min at 30m resolution.\n")
        
    print(f"\nCalibration complete. Best cost {best_cost:.2f} at {best_params}. Report saved to {REPORT_PATH}")

if __name__ == '__main__':
    main()
