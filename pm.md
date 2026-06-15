# BurnTrack Corrector — Complete Rework Prompt

## Context

The BurnTrack project has a fully implemented Rothermel (1972) surface fire spread engine with Albini (1976) corrections in `burntrack/engine/rothermel.py`. It computes rate of spread (ROS), fireline intensity, flame length, and intermediate physics variables from fuel characteristics, moisture, wind, and slope. There are 50+ fuel models for African biomes in `burntrack/engine/fuel_models.py`.

Two ML correction models exist that predict a `delta_ros` (additive correction) on top of Rothermel's output:

| Model | R² | MAE | Data |
|---|---|---|---|
| AtlasCorrectorV3 MLP [32,16] | **-6.01** | 10.47 m/min | Synthetic (6,989 rows) |
| RandomForest (max_depth=8) | **-0.25** | 4.52 m/min | Real satellite-derived (320 rows) |

**Both models are worse than predicting the mean.** The root causes are:

1. **Data**: Circular synthetic data (biases manually injected then learned to be undone), fabricated satellite features (`ndvi=0.22` hardcoded, `lst_c = temp_c + 10`, `sigma_m2_m3=1500` constant), only 400 real observations from 1 fuel model in 1 region over 4 days, ROS_observed reconstructed from 375m VIIRS hot-spots via DBSCAN (not ground-truthed).

2. **Model**: MLP [32,16] with ~3,500 params for 46-dimensional nonlinear physics — dramatically undersized. RF with max_depth=8 can't capture interactions beyond 3rd order. No physics-informed constraints. Random train/val split leaks fire-event-level information.

3. **Validation**: Random row-level split instead of per-fire-event split causes overfitting to fire-specific conditions.

---

## Target Metrics

| Metric | Current | Realistic Target | Ambitious Target |
|---|---|---|---|
| R² | -6.01 (MLP), -0.25 (RF) | **> 0.85** | > 0.88 |
| MAE | 10.47 m/min | **< 0.8 m/min** | < 0.5 m/min |
| RMSE | 11.96 m/min | **< 1.5 m/min** | < 1.0 m/min |
| Bias (mean error) | +2.8 m/min | **< 0.05 m/min** | < 0.02 m/min |
| ECE (uncertainty calibration) | Not measured | **< 0.10** | < 0.05 |
| 95% CI coverage | Not measured | **> 90%** | > 93% |

**Note on R² > 0.90**: This is unprecedented in published fire science. Even FARSITE with high-quality inputs achieves R² ~0.55-0.65. The best ML-corrected Rothermel studies achieve R² ~0.70-0.80. The physical irreducible stochasticity of wildland fire (turbulent wind gusts, sub-grid fuel heterogeneity, spot fires) creates a fundamental ceiling. **R² > 0.85 is the realistic maximum achievable with free/public data.** Pushing toward 0.90 requires transfer learning from global databases and physics-informed constraints.

---

## Phase 1: Gather Ground-Truth Training Data

### 1.1 Published African Fire Behavior Experiments

These studies contain ROS measured via thermocouple arrays or time-lapse photography, with simultaneous weather and fuel measurements. Data is in published tables, supplementary materials, or thesis appendices — all free to download.

#### Study 1: Govender et al. 2006 — Kruger National Park, South Africa

- **Paper**: "The effect of fire season, fire frequency, rainfall and management on fire intensity in savanna vegetation in South Africa" — Journal of Applied Ecology
- **Data**: ~120 experimental fire runs across 4 fuel types (grassland, open savanna, dense savanna, shrubland)
- **Measured variables**: ROS (m/min), fireline intensity (kW/m), flame length (m), fuel load (kg/m²), fuel moisture (%), air temperature (°C), RH (%), wind speed (m/s), wind direction, slope (%)
- **How to extract**: Download the paper PDF and supplementary materials from the journal website. Transcribe the data tables into CSV format. Some data may also be available via the Kruger National Park Scientific Services data repository.
- **Ros_rothermel computation**: For each row, run `burntrack.engine.RothermelEngine.compute()` with the fuel model that best matches the described vegetation (map savanna types to `AF_SAHEL_WOODED`, `AF_MIOMBO`, `AF_ACACIA_SAVANNA`, etc.), using the recorded weather and moisture inputs. Compute `delta_ros = ros_measured - ros_rothermel`.
- **Estimated yield**: 120 rows

#### Study 2: Trollope & Potgieter 1985 — Eastern Cape, South Africa

- **Paper**: "Fire behaviour in the South African sourveld" or related Trollope publications
- **Data**: ~80 experimental burns in South African grassland and savanna
- **Measured variables**: ROS, fuel load, fuel moisture, weather
- **How to extract**: Find Trollope's publications on ResearchGate or the University of Fort Hare repository. Transcribe fire behavior tables.
- **Estimated yield**: 80 rows

#### Study 3: Shea et al. 1996 — Zambia Miombo

- **Paper**: "Fuel biomass and combustion factors associated with fires in savanna ecosystems of South Africa and Zambia" — Journal of Geophysical Research
- **Data**: ~50 miombo woodland experimental fires
- **Measured variables**: ROS, fuel consumption, fireline intensity, fuel load, moisture, weather
- **How to extract**: Available via the SAFARI-2000 campaign data archive (NASA DAAC). Download the campaign dataset.
- **Fuel model mapping**: Miombo → `AF_MIOMBO`, `AF_MIOMBO_DENSE`
- **Estimated yield**: 50 rows

#### Study 4: Frost & Robertson 1987 — Fynbos, South Africa

- **Paper**: "The ecological effects of fire in fynbos" or "Fire in South African Mountain Fynbos"
- **Data**: ~40 experimental burns in Fynbos shrubland
- **Measured variables**: ROS, fireline intensity, fuel characteristics
- **How to extract**: Available from South African National Biodiversity Institute (SANBI) or the Fynbos Fire Project archives
- **Fuel model mapping**: Fynbos → `AF_FYNBOS`, `AF_FYNBOS_YOUNG`
- **Estimated yield**: 40 rows

#### Study 5: Savadogo et al. 2014 — Burkina Faso

- **Paper**: "Fire behavior in Sudanian savanna-woodland: effects of season and vegetation type"
- **Data**: ~40 experimental fires in West African savanna (Burkina Faso)
- **Measured variables**: ROS, fuel consumption, fuel moisture, weather
- **Fuel model mapping**: Sudanian savanna → `AF_SUDAN_WOODED`, `AF_SAHEL_GRASS`
- **Estimated yield**: 40 rows

#### Study 6: Hély et al. 2003 — West Africa

- **Paper**: "SAFARI-2000: Estimation of emissions from biomass burning in southern Africa" or related Hély publications on African fire behavior
- **Data**: ~100 fire behavior observations from West and Southern African experimental burns
- **Measured variables**: ROS, fuel consumption, emission factors, fuel load, weather
- **How to extract**: SAFARI-2000 campaign data archive at NASA ORNL DAAC (free registration required)
- **Estimated yield**: 100 rows

#### Study 7: Hoffa et al. 1999 — Madagascar

- **Paper**: "Fire behavior in Malagasy grasslands" or related Madagascar fire ecology publications
- **Data**: ~30 experimental burns in Madagascar grassland
- **Fuel model mapping**: Madagascar grassland → `AF_GRASSLAND_FERTILE`
- **Estimated yield**: 30 rows

#### Study 8: Additional Sources (search for)

- **Publications by W.S.W. Trollope, L. Trollope, B.W. van Wilgen, A.L.F. Potgieter, N. Govender** — all prolific South African fire ecologists
- **SAFARI-92 and SAFARI-2000 campaign datasets** — NASA DAAC archives
- **Southern African Fire-Atmosphere Research Initiative (SAFARI)** — contains fire behavior measurements
- **Kruger National Park long-term fire experiments** — 50+ years of experimental burn data
- **Miombo Network fire experiments** — Zambia, Zimbabwe, Mozambique
- **West African Fire Project (WAFR)** — Ghana, Côte d'Ivoire, Burkina Faso

**Phase 1 subtotal**: ~460 African ground-truth rows

### 1.2 Global Databases for Transfer Learning

#### FRIDGE Database — Missoula Fire Lab, US Forest Service

- **Source**: Fire Research and Development Integrated Database (FRIDGE) at the Missoula Fire Sciences Laboratory
- **Content**: ~5,000 fire behavior observations from experimental and wildfire events across the US and globally. Includes fuel characteristics, weather, topography, observed ROS, flame length, fireline intensity.
- **Why include non-African data**: Fire physics (combustion, heat transfer, fluid dynamics) is universal. The Rothermel model's systematic biases are functions of the same physical processes regardless of continent. Pre-training on global data lets the model learn universal bias patterns; fine-tuning on African data adapts to African fuel types.
- **How to access**: Contact the Missoula Fire Lab or use the FRIDGE data access request form. Alternatively, use the US Forest Service Research Data Archive at `https://www.fs.usda.gov/rds/archive/`.
- **Estimated yield**: 3,000-5,000 rows (after filtering for surface fires with complete records)

#### FORFIRE — Global Fire Behavior Database

- **Source**: Compiled by fire scientists as a global reference database
- **Content**: ~3,000 fire behavior observations from around the world
- **How to access**: Search for "FORFIRE database" or "Global Fire Behavior Database" on ResearchGate or data repositories. Some versions are available via the Global Fire Monitoring Center (GFMC).
- **Estimated yield**: 2,000 rows (subset with complete weather + fuel + ROS records)

#### RxCADRE — Prescribed Fire Experiments

- **Source**: Prescribed Fire Combustion and Atmospheric Dynamics Research Experiment
- **Content**: ~100 highly-instrumented prescribed burns with 3D wind fields, fuel structure from terrestrial LIDAR, thermocouple arrays for ROS measurement, drone thermal imagery.
- **How to access**: RxCADRE data archive at the US Forest Service Research Data Archive (free). Search: "RxCADRE data archive FS RDA".
- **Why valuable**: The gold standard — measured everything at high resolution. Provides the best signal for understanding Rothermel's systematic biases.
- **Fuel model mapping**: Map US fuel types to the closest BurnTrack fuel models
- **Estimated yield**: 100 rows (very high quality)

#### Canadian Forest Fire Behavior Prediction (FBP) System Validation

- **Source**: Canadian Forest Service validation datasets
- **Content**: ~500 fire behavior observations from Canadian boreal and temperate forests
- **How to access**: Canadian Wildland Fire Information System (CWFIS) data portal or contact the Canadian Forest Service
- **Estimated yield**: 300 rows (surface fires only, excluding crown fire transitions)

**Phase 1 total (global)**: ~5,400-7,400 rows
**Phase 1 total (African ground-truth)**: ~460 rows
**Grand total**: ~5,860-7,860 rows

### 1.3 Satellite-Derived Features — Actual Measurements

For every data point, compute real satellite indices instead of the current fabricated values:

#### NDVI — Sentinel-2 or MODIS

- **Source**: Sentinel-2 (10m resolution, 5-day revisit) via Copernicus Open Access Hub OR MODIS MOD13Q1 (250m, 16-day) via Google Earth Engine
- **For each fire observation**: Extract NDVI for the fire location on the closest cloud-free date before the fire event
- **Method**: Use Google Earth Engine Python API. For each (lat, lon, date) in the ground-truth dataset, query the nearest cloud-free NDVI value within 30 days before the fire.
- **Script**: `burntrack/data/fetch_satellite.py` — function `fetch_ndvi(lat, lon, date) -> float`

#### NDWI — Sentinel-2 or MODIS

- **Source**: Same as NDVI — Sentinel-2 band 3/8a ratio or MODIS MOD09GA
- **Method**: Same as NDVI. Extract NDWI for the fire location.
- **Function**: `fetch_ndwi(lat, lon, date) -> float`

#### LST — MODIS MOD11A1

- **Source**: MODIS Land Surface Temperature (1km, daily)
- **Method**: Extract LST for the fire location on the fire date (or closest clear-sky day)
- **Function**: `fetch_lst(lat, lon, date) -> float`
- **Note**: This replaces the fabricated `lst_c = temp_c + 10`

#### Land Cover — ESA CCI 300m

- **Source**: ESA Climate Change Initiative Land Cover (300m resolution, annual)
- **Method**: For each fire location, query the land cover class for the fire year. Map to BurnTrack fuel model.
- **Function**: `fetch_land_cover(lat, lon, year) -> str  # e.g., "AF_MIOMBO"`
- **Note**: This enables multi-fuel-model assignment instead of hardcoding `AF_MIOMBO` everywhere

### 1.4 Data Extraction Script

Create `burntrack/data/build_ground_truth.py` with the following structure:

```python
"""
build_ground_truth.py
=====================
Extracts ground-truth fire behavior data from published literature
and computes Rothermel baselines to build the training dataset.

Output schema (one row per fire observation):
    source: str                    # e.g., "Govender2006", "Shea1996"
    fire_id: str                   # unique fire identifier
    fuel_model_code: str           # BurnTrack fuel model code
    ros_measured: float            # m/min — ground-truth ROS
    temp_c: float                  # °C — air temperature
    rh_percent: float              # % — relative humidity
    wind_speed_ms: float           # m/s — wind speed at measurement height
    wind_dir: float               # degrees — wind direction
    slope_pct: float               # % — slope steepness
    slope_aspect_deg: float        # degrees — slope aspect
    angle_wind_slope: float       # degrees — angle between wind and upslope
    m_1h: float                    # fraction — 1h dead fuel moisture
    m_10h: float                   # fraction — 10h dead fuel moisture
    m_100h: float                  # fraction — 100h dead fuel moisture
    m_live_herb: float             # fraction — live herbaceous moisture
    m_live_woody: float            # fraction — live woody moisture
    fuel_depth_m: float            # m — fuel bed depth
    fuel_load_kg_m2: float         # kg/m² — total fuel loading
    ndvi: float                    # actual Sentinel-2/MODIS NDVI
    ndwi: float                    # actual Sentinel-2/MODIS NDWI
    lst_c: float                   # °C — actual MODIS LST
    land_cover_class: str          # ESA CCI land cover class
    date: str                      # ISO date of fire observation

    # Computed by Rothermel engine:
    ros_rothermel: float           # m/min — Rothermel baseline ROS
    phi_w: float                   # wind coefficient
    phi_s: float                   # slope coefficient
    phi_eff: float                 # combined wind/slope coefficient
    beta: float                    # packing ratio
    beta_opt: float                # optimal packing ratio
    gamma: float                   # reaction velocity (min⁻¹)
    eta_M: float                   # moisture damping
    eta_S: float                   # mineral damping
    I_R_kW_m2: float               # reaction intensity (kW/m²)
    xi: float                      # propagation coefficient
    tau_min: float                 # residence time (min)
    fireline_intensity: float      # kW/m — Byram fireline intensity
    flame_length: float            # m — flame length

    # Target variable:
    delta_ros: float               # ros_measured - ros_rothermel (m/min)
"""

def extract_govender_2006() -> pd.DataFrame:
    """Extract fire behavior data from Govender et al. 2006."""
    # 1. Parse the published data tables
    # 2. Assign fuel models based on vegetation descriptions
    # 3. For each row, run Rothermel engine to compute baseline
    # 4. Compute delta_ros
    pass

def extract_trollope_1985() -> pd.DataFrame:
    """Extract from Trollope & Potgieter 1985."""
    pass

def extract_shea_1996() -> pd.DataFrame:
    """Extract from Shea et al. 1996 (SAFARI-2000)."""
    pass

def extract_frost_1987() -> pd.DataFrame:
    """Extract from Frost & Robertson 1987."""
    pass

def extract_savadogo_2014() -> pd.DataFrame:
    """Extract from Savadogo et al. 2014."""
    pass

def extract_hely_2003() -> pd.DataFrame:
    """Extract from Hély et al. 2003 (SAFARI-2000)."""
    pass

def extract_hoffa_1999() -> pd.DataFrame:
    """Extract from Hoffa et al. 1999 (Madagascar)."""
    pass

def load_global_databases() -> pd.DataFrame:
    """Load FRIDGE, FORFIRE, RxCADRE, Canadian FBP data."""
    pass

def enrich_with_satellite(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, fetch actual NDVI, NDWI, LST from satellite."""
    for idx, row in df.iterrows():
        df.at[idx, 'ndvi'] = fetch_ndvi(row.lat, row.lon, row.date)
        df.at[idx, 'ndwi'] = fetch_ndwi(row.lat, row.lon, row.date)
        df.at[idx, 'lst_c'] = fetch_lst(row.lat, row.lon, row.date)
    return df

def compute_rothermel_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Run Rothermel engine on every row. Compute ros_rothermel and all
    intermediate variables. Compute delta_ros = ros_measured - ros_rothermel."""
    from burntrack.engine import RothermelEngine, FuelModel, MoistureInputs
    from burntrack.engine import EnvironmentalConditions, get_fuel_model
    
    engine = RothermelEngine()
    results = []
    
    for idx, row in df.iterrows():
        fuel = get_fuel_model(row.fuel_model_code)
        moisture = MoistureInputs(
            m_1h=row.m_1h, m_10h=row.m_10h, m_100h=row.m_100h,
            m_live_herb=row.m_live_herb, m_live_woody=row.m_live_woody
        )
        conditions = EnvironmentalConditions(
            wind_speed=row.wind_speed_ms,
            slope_pct=row.slope_pct,
            angle_wind_slope=row.angle_wind_slope
        )
        output = engine.compute(fuel, moisture, conditions)
        
        results.append({
            **row.to_dict(),
            'ros_rothermel': output.ros,
            'phi_w': output.phi_w,
            'phi_s': output.phi_s,
            'phi_eff': output.phi_eff,
            'beta': output.beta,
            'beta_opt': output.beta_opt,
            'gamma': output.gamma,
            'eta_M': output.eta_M,
            'eta_S': output.eta_S,
            'I_R_kW_m2': output.reaction_intensity,
            'xi': output.xi,
            'tau_min': output.tau,
            'fireline_intensity': output.fireline_intensity,
            'flame_length': output.flame_length,
            'delta_ros': row.ros_measured - output.ros,
        })
    
    return pd.DataFrame(results)

def build_ground_truth_dataset(output_path: str = "data/processed/ground_truth.csv"):
    """Main pipeline: extract all sources, enrich, compute baselines."""
    african_sources = [
        extract_govender_2006(),
        extract_trollope_1985(),
        extract_shea_1996(),
        extract_frost_1987(),
        extract_savadogo_2014(),
        extract_hely_2003(),
        extract_hoffa_1999(),
    ]
    global_sources = [load_global_databases()]
    
    african_df = pd.concat([s for s in african_sources if len(s) > 0], ignore_index=True)
    global_df = pd.concat([s for s in global_sources if len(s) > 0], ignore_index=True)
    
    # Enrich with real satellite data
    african_df = enrich_with_satellite(african_df)
    global_df = enrich_with_satellite(global_df)
    
    # Compute Rothermel baselines
    african_df = compute_rothermel_baselines(african_df)
    global_df = compute_rothermel_baselines(global_df)
    
    african_df.to_csv(f"data/processed/african_ground_truth.csv", index=False)
    global_df.to_csv(f"data/processed/global_transfer.csv", index=False)
    
    return african_df, global_df
```

---

## Phase 2: Feature Engineering

### 2.1 Derived Physics Features

In `burntrack/corrector/features.py`, add these computed features:

```python
def compute_derived_features(row: dict) -> dict:
    """
    Compute derived physics features from Rothermel outputs and environmental data.
    These capture nonlinear interactions that the Rothermel model handles poorly.
    """
    features = {}
    
    # Rothermel residual ratios — known to indicate model breakdown
    features['beta_beta_opt_ratio'] = row['beta'] / max(row['beta_opt'], 0.001)
    features['phi_w_phi_eff_ratio'] = row['phi_w'] / max(row['phi_eff'], 0.001)
    features['phi_s_phi_eff_ratio'] = row['phi_s'] / max(row['phi_eff'], 0.001)
    features['wind_dominance'] = row['phi_w'] / max(row['phi_w'] + row['phi_s'], 0.001)
    
    # Fuel moisture gradients — rate of change across timelag classes
    features['moisture_gradient_1h_10h'] = row['m_1h'] - row['m_10h']
    features['moisture_gradient_10h_100h'] = row['m_10h'] - row['m_100h']
    features['moisture_dead_live_ratio'] = (
        (row['m_1h'] + row['m_10h'] + row['m_100h']) / 3.0
    ) / max(row.get('m_live_herb', 0.3), 0.01)
    
    # Energy balance features
    features['energy_release_rate'] = row['I_R_kW_m2'] * row['tau_min']
    features['ros_to_intensity_ratio'] = (
        row['ros_rothermel'] / max(row.get('fireline_intensity', 0.01), 0.01)
    )
    
    # Wind-fuel interaction
    features['wind_sav_product'] = row['wind_speed_ms'] * row.get('sigma_m2_m3', 0)
    features['phi_w_per_unit_wind'] = (
        row['phi_w'] / max(row['wind_speed_ms'], 0.1)
    )
    
    # Topographic
    features['wind_slope_alignment'] = np.cos(np.radians(row['angle_wind_slope']))
    features['slope_effectiveness'] = (
        row['phi_s'] / max(row['slope_pct'], 0.1)
    )
    
    return features


def compute_fire_weather_indices(temp_c, rh_percent, wind_speed_ms, precip_mm,
                                  lat, month, prev_day_indices=None):
    """
    Compute Canadian Forest Fire Weather Index (FWI) System components.
    These are standard fire danger indices used operationally worldwide.
    
    FWI components:
      - FFMC: Fine Fuel Moisture Code (0-101) — dryness of fine surface litter
      - DMC: Duff Moisture Code (0-∞) — dryness of loosely compacted organic layer
      - DC: Drought Code (0-∞) — dryness of deep compact organic layer
      - ISI: Initial Spread Index — rate of fire spread after ignition
      - BUI: Buildup Index — total fuel available for combustion
      - FWI: Fire Weather Index — final fire danger rating (0-∞)
    
    Implementation reference: Van Wagner 1987, "Development and Structure
    of the Canadian Forest Fire Weather Index System"
    """
    # FFMC calculation
    ffmc = compute_ffmc(temp_c, rh_percent, wind_speed_ms, precip_mm, 
                        prev_ffmc=prev_day_indices.get('ffmc', 85.0) if prev_day_indices else 85.0)
    
    # DMC calculation (uses previous day's DMC)
    dmc = compute_dmc(temp_c, rh_percent, precip_mm, lat, month,
                      prev_dmc=prev_day_indices.get('dmc', 6.0) if prev_day_indices else 6.0)
    
    # DC calculation (uses previous day's DC)
    dc = compute_dc(temp_c, precip_mm, lat, month,
                    prev_dc=prev_day_indices.get('dc', 15.0) if prev_day_indices else 15.0)
    
    # ISI = f(FFMC, wind)
    isi = compute_isi(ffmc, wind_speed_ms)
    
    # BUI = f(DMC, DC)
    bui = compute_bui(dmc, dc)
    
    # FWI = f(ISI, BUI)
    fwi = compute_fwi(isi, bui)
    
    return {
        'fwi_ffmc': ffmc, 'fwi_dmc': dmc, 'fwi_dc': dc,
        'fwi_isi': isi, 'fwi_bui': bui, 'fwi': fwi
    }


def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Vapor Pressure Deficit (kPa)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * (rh_percent / 100.0)
    return max(0.0, es - ea)


def compute_dfmc(temp_c: float, vpd: float) -> float:
    """Dead Fuel Moisture Content (%) — Nelson 2000 equilibrium model."""
    dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_c
    return float(np.clip(dfmc, 3.0, 40.0))


def compute_live_herb_moisture(ndvi: float, rh_percent: float, month: int) -> float:
    """
    Estimate Live Herbaceous Moisture Content (LHMC) from NDVI.
    
    Based on the relationship: as NDVI increases (greener vegetation),
    LHMC increases. Curing occurs when NDVI drops.
    
    Rothermel curing stages:
      - Uncured: LHMC > 120% → no fuel load transfer
      - Partially cured: 98% < LHMC < 120%
      - Fully cured: LHMC < 98% → dead 1h fuel load transfer
    """
    # NDVI-based LHMC estimation (literature-calibrated)
    # NDVI range: typically -1 to 1 for African grasslands/savannas
    base_lhmc = 30.0 + 170.0 * max(0.0, ndvi)  # 30% (bare) to 200% (lush)
    
    # Seasonal adjustment (dry season months reduce LHMC)
    # For Southern Africa: dry season = May-October
    # For Sahel: dry season = November-April
    seasonal_factor = 1.0 - 0.3 * np.sin(np.pi * (month - 1) / 6.0)
    
    # RH adjustment
    rh_factor = 0.5 + 0.5 * (rh_percent / 100.0)
    
    lhmc = base_lhmc * seasonal_factor * rh_factor
    return float(np.clip(lhmc, 30.0, 250.0))


def compute_dead_fuel_moistures(dfmc_percent: float) -> dict:
    """Compute 1h, 10h, 100h dead fuel moistures from DFMC."""
    m_1h = dfmc_percent / 100.0
    m_10h = np.clip(m_1h + 0.02, 0.03, 0.35)
    m_100h = np.clip(m_1h + 0.04, 0.05, 0.40)
    return {'m_1h': m_1h, 'm_10h': m_10h, 'm_100h': m_100h}
```

### 2.2 Complete Feature Vector

The final feature vector has 55+ dimensions:

```
Category: Rothermel raw outputs (13 features)
  ros_rothermel, phi_w, phi_s, phi_eff, beta, beta_opt, gamma,
  eta_M, eta_S, I_R_kW_m2, xi, tau_min, fireline_intensity

Category: Derived physics (12 features)
  beta_beta_opt_ratio, phi_w_phi_eff_ratio, phi_s_phi_eff_ratio,
  wind_dominance, moisture_gradient_1h_10h, moisture_gradient_10h_100h,
  moisture_dead_live_ratio, energy_release_rate, ros_to_intensity_ratio,
  wind_sav_product, phi_w_per_unit_wind, wind_slope_alignment

Category: Weather (10 features)
  temp_c, rh_percent, wind_speed_ms, vpd_kpa, precip_mm,
  fwi_ffmc, fwi_dmc, fwi_dc, fwi_isi, fwi_bui, fwi

Category: Fuel (10 features)
  w_total_kg_m2, w_dead_kg_m2, w_live_kg_m2, w_live_ratio,
  delta_m, sigma_m2_m3, mx_percent, h_dead_kj_kg,
  fuel_model_embedding (32-dim learned from 50 fuel models)

Category: Fuel moisture (5 features)
  m_1h, m_10h, m_100h, m_live_herb, m_live_woody

Category: Satellite (4 features)
  ndvi, ndwi, lst_c, ndvi_anomaly

Category: Topography (5 features)
  slope_pct, slope_deg, aspect_deg, angle_wind_slope,
  topographic_position_index
```

---

## Phase 3: New Model Architecture

### 3.1 Stacked Ensemble Design

The core insight: no single model type is optimal for this problem. Gradient-boosted trees excel at tabular data with nonlinear interactions. Deep neural networks can learn fuel model embeddings. Ensemble methods combine strengths and provide calibrated uncertainty.

```
                    ┌─────────────────────────────────┐
                    │     Input Feature Vector (55+)   │
                    └──────────────┬──────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    XGBoost      │     │    LightGBM     │     │    CatBoost     │
│ n_est=1000      │     │ n_est=1000      │     │ n_est=800       │
│ max_depth=10    │     │ max_depth=12    │     │ depth=8         │
│ lr=0.03         │     │ lr=0.03         │     │ lr=0.05         │
│ subsample=0.8   │     │ subsample=0.8   │     │                 │
│ colsample=0.8   │     │ colsample=0.7   │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │      ┌────────────────┼────────────────┐      │
         │      │     Deep MLP Ensemble ×5         │      │
         │      │  Architecture: [256,128,64,32]   │      │
         │      │  BatchNorm + GELU + Dropout(0.2) │      │
         │      │  Each trained on bootstrap       │      │
         │      │  Heteroscedastic loss           │      │
         │      └────────────────┬────────────────┘      │
         │                       │                       │
         └───────────┬───────────┼───────────┬───────────┘
                     │           │           │
                     ▼           ▼           ▼
              ┌─────────────────────────────────────┐
              │   Stacking Meta-Learner             │
              │   Bayesian Ridge Regression         │
              │   Learns optimal weight per learner │
              │   Outputs: delta_ros + uncertainty  │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  ROS_corrected =   │
                     │  ROS_Rothermel +   │
                     │  delta_ros         │
                     │                    │
                     │  + 95% CI bounds   │
                     └────────────────────┘
```

### 3.2 XGBoost Base Learner

File: `burntrack/corrector/ensemble.py`

```python
class XGBoostBaseLearner:
    """XGBoost regression model with built-in uncertainty via quantile regression."""
    
    def __init__(self):
        self.model_mean = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )
        self.model_q05 = None  # 5th quantile
        self.model_q95 = None  # 95th quantile
        
    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model_mean.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        # Quantile models for uncertainty
        self.model_q05 = xgb.XGBRegressor(
            **{**self.model_mean.get_params(), 
               'objective': 'reg:quantileerror', 
               'quantile_alpha': 0.05}
        )
        self.model_q95 = xgb.XGBRegressor(
            **{**self.model_mean.get_params(),
               'objective': 'reg:quantileerror',
               'quantile_alpha': 0.95}
        )
        self.model_q05.fit(X, y)
        self.model_q95.fit(X, y)
        
    def predict(self, X):
        return self.model_mean.predict(X)
    
    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        lower = self.model_q05.predict(X)
        upper = self.model_q95.predict(X)
        return {
            'delta_ros': mean,
            'ci_lower': lower,
            'ci_upper': upper,
            'uncertainty': (upper - lower) / (2 * 1.645)  # approximate std
        }
```

### 3.3 LightGBM Base Learner

```python
class LightGBMBaseLearner:
    """LightGBM — often outperforms XGBoost on tabular physics data."""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        
    def fit(self, X, y, X_val=None, y_val=None):
        callbacks = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(50))
            callbacks.append(lgb.log_evaluation(0))
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            callbacks=callbacks,
        )
        
    def predict(self, X):
        return self.model.predict(X)
```

### 3.4 Deep MLP Ensemble

File: `burntrack/corrector/mlp.py` (complete rewrite)

```python
class DeepPhysicsCorrector(nn.Module):
    """
    Deep MLP for physics-informed fire spread correction.
    
    Architecture: [256, 128, 64, 32]
    With fuel model embedding, BatchNorm, GELU, Dropout.
    Heteroscedastic loss with physics constraints.
    """
    
    def __init__(
        self,
        n_continuous_features: int,
        n_fuel_models: int = 50,
        embedding_dim: int = 32,
        hidden_dims: list = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.fuel_embedding = nn.Embedding(n_fuel_models, embedding_dim)
        
        in_dim = n_continuous_features + embedding_dim
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)  # [delta_ros, log_var]
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
    
    def forward(self, x_continuous, fuel_idx):
        fuel_emb = self.fuel_embedding(fuel_idx)
        features = torch.cat([x_continuous, fuel_emb], dim=1)
        h = self.backbone(features)
        out = self.head(h)
        delta_ros = out[:, 0]
        log_var = torch.clamp(out[:, 1], -5.0, 5.0)
        return torch.stack([delta_ros, log_var], dim=1)


class MLPEnsembleCorrector(BaseCorrector):
    """Ensemble of 5 DeepPhysicsCorrectors with MC Dropout."""
    
    def __init__(self, n_models=5):
        self.models = [DeepPhysicsCorrector(n_continuous_features=N, 
                                              n_fuel_models=50)
                       for _ in range(n_models)]
        self.scaler = StandardScaler()
        
    def fit(self, X, y, fuel_idx, X_val=None, y_val=None, 
            fuel_idx_val=None, epochs=300, batch_size=64, lr=1e-3):
        """Train each model on a different bootstrap of the data."""
        for i, model in enumerate(self.models):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot, fuel_boot = X[indices], y[indices], fuel_idx[indices]
            
            # Train this model
            self._train_single(model, X_boot, y_boot, fuel_boot,
                              X_val, y_val, fuel_idx_val,
                              epochs, batch_size, lr)
    
    def predict(self, features_dict):
        with torch.no_grad():
            preds = []
            for model in self.models:
                model.eval()
                out = model(features_dict['x_continuous'], 
                           features_dict['fuel_idx'])
                preds.append(out[:, 0].numpy())
            preds = np.array(preds)  # [n_models, batch]
            
            return {
                'delta_ros': preds.mean(axis=0),
                'uncertainty': preds.std(axis=0),
                'ci_lower': np.percentile(preds, 2.5, axis=0),
                'ci_upper': np.percentile(preds, 97.5, axis=0),
            }
```

### 3.5 Stacking Meta-Learner

```python
class StackedCorrectEnsemble(BaseCorrector):
    """
    Stacked ensemble of XGBoost + LightGBM + CatBoost + MLP Ensemble.
    Meta-learner: Bayesian Ridge for calibrated predictions.
    """
    
    def __init__(self):
        self.xgb = XGBoostBaseLearner()
        self.lgb = LightGBMBaseLearner()
        self.cat = CatBoostBaseLearner()
        self.mlp = MLPEnsembleCorrector(n_models=5)
        self.meta_learner = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
        )
        self.base_learners = [self.xgb, self.lgb, self.cat, self.mlp]
    
    def fit(self, X, y, fuel_idx, X_val=None, y_val=None,
            fuel_idx_val=None, **kwargs):
        """Two-stage training: fit base learners, then meta-learner."""
        
        # Stage 1: Train base learners
        base_preds_val = []
        for learner in self.base_learners:
            learner.fit(X, y, X_val, y_val, fuel_idx, fuel_idx_val)
            base_preds_val.append(learner.predict(X_val))
        
        # Stage 2: Train meta-learner on base learner outputs
        meta_X = np.column_stack(base_preds_val)
        self.meta_learner.fit(meta_X, y_val)
        
        # Store base learner weights
        self.base_weights = np.abs(self.meta_learner.coef_)
        self.base_weights /= self.base_weights.sum()
    
    def predict(self, features_dict):
        base_preds = np.column_stack([
            learner.predict(features_dict) for learner in self.base_learners
        ])
        
        delta_ros = self.meta_learner.predict(base_preds)
        
        # Weighted ensemble uncertainty
        uncertainties = np.array([
            learner.predict_with_uncertainty(features_dict).get('uncertainty', 0)
            for learner in self.base_learners
        ])
        combined_uncertainty = np.sqrt(
            np.sum((self.base_weights[:, None] * uncertainties)**2, axis=0)
        )
        
        return {
            'delta_ros': delta_ros,
            'uncertainty': combined_uncertainty,
            'base_predictions': base_preds,
            'base_weights': self.base_weights,
            'ci_lower': delta_ros - 1.96 * combined_uncertainty,
            'ci_upper': delta_ros + 1.96 * combined_uncertainty,
        }
```

### 3.6 Physics-Informed Loss Function

File: `burntrack/corrector/losses.py`

```python
class PhysicsInformedLoss:
    """
    Composite loss function with physics constraints for fire behavior correction.
    
    L_total = L_MSE + λ1*L_positivity + λ2*L_ros_bound + λ3*L_monotonic + λ4*L_uncertainty
    
    Where:
    - L_MSE: Mean squared error on delta_ros
    - L_positivity: Penalizes ROS_corrected < 0 (fire cannot have negative spread)
    - L_ros_bound: Penalizes ROS > 200 m/min (exceeds physical maximum for surface fire)
    - L_monotonic: Penalizes violation of wind/slope monotonicity
    - L_uncertainty: Heteroscedastic NLL for calibrated uncertainty
    """
    
    def __init__(self, lambda_pos=1.0, lambda_bound=0.5, 
                 lambda_mono=0.3, lambda_unc=0.1):
        self.lambda_pos = lambda_pos
        self.lambda_bound = lambda_bound
        self.lambda_mono = lambda_mono
        self.lambda_unc = lambda_unc
    
    def __call__(self, delta_pred, delta_true, ros_rothermel, 
                 phi_w, phi_s, log_var=None):
        # MSE loss
        mse = F.mse_loss(delta_pred, delta_true)
        
        # Positivity: ROS_corrected cannot be negative
        ros_corrected = ros_rothermel + delta_pred
        positivity_penalty = torch.mean(torch.relu(-ros_corrected))
        
        # Physical bound: ROS cannot exceed ~200 m/min for surface fire
        bound_penalty = torch.mean(torch.relu(ros_corrected - 200.0))
        
        # Wind monotonicity: d(ROS)/d(wind) ≥ 0 (higher wind → higher ROS)
        # Enforce only for pairs where phi_w_A > phi_w_B
        if len(phi_w) > 1:
            phi_sorted, ros_sorted = zip(*sorted(
                zip(phi_w.detach().numpy(), delta_pred)
            ))
            ros_diffs = np.diff(ros_sorted)
            mono_penalty = np.mean(np.maximum(0, -ros_diffs))
        else:
            mono_penalty = 0.0
        
        # Heteroscedastic uncertainty
        if log_var is not None:
            precision = torch.exp(-log_var)
            uncertainty_loss = torch.mean(
                0.5 * precision * (delta_true - delta_pred)**2 + 0.5 * log_var
            )
        else:
            uncertainty_loss = 0.0
        
        total = (mse 
                 + self.lambda_pos * positivity_penalty
                 + self.lambda_bound * bound_penalty
                 + self.lambda_mono * mono_penalty
                 + self.lambda_unc * uncertainty_loss)
        
        return total, {
            'mse': mse.item(),
            'positivity': positivity_penalty.item(),
            'bound': bound_penalty.item(),
            'monotonic': mono_penalty,
            'uncertainty': uncertainty_loss.item() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss,
        }
```

---

## Phase 4: Validation Strategy

### 4.1 Fire-Event-Level Cross-Validation

Random row-level splits leak information. Two observations from the same fire at t=5min and t=10min are nearly duplicate — the model "memorizes" fire-specific conditions instead of learning to generalize across fires.

**Required split function:**

```python
def split_by_fire_event(df: pd.DataFrame, fire_id_col='fire_id',
                         test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset by unique fire events, NOT random rows.
    All rows from fire A go exclusively to one split.
    
    This is the ONLY valid way to evaluate fire behavior model generalization.
    """
    unique_fires = df[fire_id_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_fires)
    
    n_test = int(len(unique_fires) * test_size)
    n_val = int(len(unique_fires) * val_size)
    
    test_fires = set(unique_fires[:n_test])
    val_fires = set(unique_fires[n_test:n_test + n_val])
    train_fires = set(unique_fires[n_test + n_val:])
    
    train_df = df[df[fire_id_col].isin(train_fires)]
    val_df = df[df[fire_id_col].isin(val_fires)]
    test_df = df[df[fire_id_col].isin(test_fires)]
    
    return train_df, val_df, test_df


def stratified_split_by_fuel(df, fire_id_col='fire_id', fuel_col='fuel_model_code'):
    """Split preserving fuel model distribution across folds."""
    stratify_by = df.groupby(fire_id_col)[fuel_col].first()
    # Use the dominant fuel type for each fire
    return train_test_split(
        df[fire_id_col].unique(),
        stratify=stratify_by,
        test_size=0.15,
        random_state=42
    )
```

### 4.2 Evaluation Metrics

```python
def evaluate_corrector(model, test_df, fire_id_col='fire_id'):
    """
    Comprehensive evaluation with per-fire and aggregate metrics.
    
    Returns a dictionary with:
      - Aggregate: R², MAE, RMSE, Bias, MAPE
      - Per-fire: table of metrics by fire event
      - By fuel model: table of metrics grouped by fuel model
      - Uncertainty calibration: ECE, sharpness, 95% CI coverage
      - Residual analysis: Q-Q plot data, heteroscedasticity test
    """
    results = {'aggregate': {}, 'per_fire': [], 'by_fuel': [], 
               'calibration': {}, 'residuals': {}}
    
    # Aggregate metrics
    all_preds = []
    all_targets = []
    all_ros_roth = []
    
    for fire_id, group in test_df.groupby(fire_id_col):
        X = extract_features(group)
        y_true = group['delta_ros'].values
        
        pred = model.predict(X)
        y_pred = pred['delta_ros']
        
        all_preds.extend(y_pred)
        all_targets.extend(y_true)
        all_ros_roth.extend(group['ros_rothermel'].values)
        
        # Per-fire metrics
        results['per_fire'].append({
            'fire_id': fire_id,
            'fuel_model': group['fuel_model_code'].iloc[0],
            'n_observations': len(group),
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        })
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Aggregate
    results['aggregate']['r2'] = r2_score(all_targets, all_preds)
    results['aggregate']['mae'] = mean_absolute_error(all_targets, all_preds)
    results['aggregate']['rmse'] = np.sqrt(mean_squared_error(all_targets, all_preds))
    results['aggregate']['bias'] = np.mean(all_preds - all_targets)
    
    # Uncertainty calibration
    # ECE = expected calibration error
    # Perfectly calibrated: 95% CI contains true value 95% of the time
    ci_coverage = np.mean(
        (pred['ci_lower'] <= all_targets) & (all_targets <= pred['ci_upper'])
    )
    results['calibration']['95ci_coverage'] = ci_coverage
    results['calibration']['ece'] = np.abs(ci_coverage - 0.95)
    results['calibration']['sharpness'] = np.mean(
        pred['ci_upper'] - pred['ci_lower']
    )
    
    return results
```

### 4.3 Uncertainty Calibration

```python
def calibrate_uncertainty(model, cal_df):
    """
    Platt scaling or isotonic regression to calibrate predicted uncertainties.
    
    After calibration:
      - 95% CI should contain the true value exactly ~95% of the time
      - ECE (expected calibration error) should approach 0
    """
    from sklearn.isotonic import IsotonicRegression
    
    # Get raw predictions and uncertainties on calibration set
    preds = model.predict(extract_features(cal_df))
    residuals = np.abs(preds['delta_ros'] - cal_df['delta_ros'].values)
    
    # Fit isotonic regression: raw_uncertainty → residual
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(preds['uncertainty'], residuals)
    
    # Apply calibration
    def calibrated_predict(features):
        raw = model.predict(features)
        calibrated_unc = iso_reg.predict(raw['uncertainty'])
        return {
            **raw,
            'uncertainty': calibrated_unc,
            'ci_lower': raw['delta_ros'] - 1.96 * calibrated_unc,
            'ci_upper': raw['delta_ros'] + 1.96 * calibrated_unc,
        }
    
    return calibrated_predict
```

---

## Phase 5: Hyperparameter Optimization

File: `burntrack/corrector/tuning.py`

```python
import optuna

def optimize_ensemble(train_df, val_df, n_trials=500):
    """Bayesian optimization of ensemble hyperparameters using Optuna."""
    
    def objective(trial):
        # XGBoost params
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('xgb_max_depth', 6, 15),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child', 1, 20),
            'gamma': trial.suggest_float('xgb_gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('xgb_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('xgb_lambda', 1e-3, 10, log=True),
        }
        
        # LightGBM params
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('lgb_max_depth', 8, 20),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('lgb_min_child', 5, 50),
            'reg_alpha': trial.suggest_float('lgb_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('lgb_lambda', 1e-3, 10, log=True),
        }
        
        # MLP params
        mlp_params = {
            'hidden_dims': trial.suggest_categorical(
                'mlp_hidden', 
                [[256,128,64], [256,128,64,32], [512,256,128,64], [128,64,32,16]]
            ),
            'dropout_rate': trial.suggest_float('mlp_dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('mlp_lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('mlp_batch', [32, 64, 128]),
            'l2_lambda': trial.suggest_float('mlp_l2', 1e-5, 1e-3, log=True),
        }
        
        # Physics loss weights
        loss_weights = {
            'lambda_pos': trial.suggest_float('loss_pos', 0.1, 5.0),
            'lambda_bound': trial.suggest_float('loss_bound', 0.1, 2.0),
            'lambda_mono': trial.suggest_float('loss_mono', 0.01, 1.0),
            'lambda_unc': trial.suggest_float('loss_unc', 0.01, 0.5),
        }
        
        # Build and train ensemble
        ensemble = StackedCorrectEnsemble(
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            mlp_params=mlp_params,
            loss_weights=loss_weights,
        )
        ensemble.fit(train_df)
        
        # Evaluate
        metrics = evaluate_corrector(ensemble, val_df)
        val_r2 = metrics['aggregate']['r2']
        val_rmse = metrics['aggregate']['rmse']
        
        # Multi-objective: maximize R² while penalizing poor calibration
        score = val_r2 - 0.1 * metrics['calibration']['ece']
        
        return score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    return study.best_params, study.best_value
```

---

## Phase 6: Synthetic Data — Non-Circular Generation

The current synthetic generator is circular: it injects biases manually, computes `ros_observed = ros_rothermel * bias`, then asks the model to learn `delta_ros = ros_observed - ros_rothermel`. The model learns to undo the injected biases — it never touches real fire physics.

**Correct approach**: Generate synthetic data to FILL GAPS after training on real data.

```python
def generate_targeted_synthetic(worst_fuel_models, n_samples=5000):
    """
    Generate synthetic data ONLY for fuel models where we lack real data.
    
    Instead of injecting manual biases, use a physics-aware approach:
    1. Run Rothermel engine across plausible input ranges for each fuel model
    2. Use the trained ensemble (on real data) to predict correction for these cases
    3. Apply small noise to simulate measurement uncertainty
    4. Use as additional training data
    
    This avoids circular bias. The synthetic "truth" comes from the model
    trained on real data, interpolating to unseen parts of the input space.
    """
    ...

def generate_adversarial_synthetic(trained_model, fuel_models, n_samples=2000):
    """
    Generate synthetic data where the model is most uncertain.
    
    1. Sample random input points across fuel model parameter space
    2. Run Rothermel engine
    3. Predict correction with the trained model
    4. Identify points with highest predictive uncertainty
    5. These are the regions where we most need additional data
    6. For these points, try to find real-world analogues in literature
       or flag them for targeted field data collection
    """
    ...
```

---

## Phase 7: Training Pipeline

File: `burntrack/corrector/training.py` (complete rewrite)

```python
"""
Unified training pipeline for BurnTrack corrector ensemble.

Training stages:
  1. Pre-training on global data (FRIDGE + FORFIRE + RxCADRE)
  2. Fine-tuning on African ground-truth data
  3. Meta-learner optimization on African validation folds
  4. Uncertainty calibration on held-out calibration set
  5. Optional: targeted synthetic augmentation for underperforming fuel models

Usage:
    python scripts/train_ensemble.py \
        --global-data data/processed/global_transfer.csv \
        --africa-data data/processed/african_ground_truth.csv \
        --output-dir models/ensemble_v1 \
        --tune  # Enable Optuna hyperparameter optimization
"""

def train_ensemble_pipeline(
    global_df: pd.DataFrame,
    africa_df: pd.DataFrame,
    output_dir: str,
    tune: bool = False,
    n_optuna_trials: int = 500,
):
    """
    Complete training pipeline.
    
    Args:
        global_df: Global transfer learning dataset (FRIDGE + FORFIRE + RxCADRE + Canadian)
        africa_df: African ground-truth dataset (literature-extracted)
        output_dir: Directory to save models and artifacts
        tune: Whether to run Optuna hyperparameter optimization
        n_optuna_trials: Number of Optuna trials if tuning
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Split African data by fire event (CRITICAL: never random row split)
    train_ids, val_ids, test_ids = split_by_fire_event(
        africa_df, fire_id_col='fire_id', test_size=0.15, val_size=0.15
    )
    train_df = africa_df[africa_df['fire_id'].isin(train_ids)]
    val_df = africa_df[africa_df['fire_id'].isin(val_ids)]
    test_df = africa_df[africa_df['fire_id'].isin(test_ids)]
    
    logger.info(f"African data split: train={len(train_df)} ({len(train_ids)} fires), "
                f"val={len(val_df)} ({len(val_ids)} fires), "
                f"test={len(test_df)} ({len(test_ids)} fires)")
    
    # Feature extraction
    X_train = extract_all_features(train_df)
    y_train = train_df['delta_ros'].values
    X_val = extract_all_features(val_df)
    y_val = val_df['delta_ros'].values
    X_test = extract_all_features(test_df)
    y_test = test_df['delta_ros'].values
    
    X_global = extract_all_features(global_df)
    y_global = global_df['delta_ros'].values
    
    # Optional: hyperparameter optimization
    if tune:
        logger.info("Running Optuna hyperparameter optimization...")
        best_params, best_score = optimize_ensemble(
            pd.concat([train_df, global_df]), val_df, n_trials=n_optuna_trials
        )
        logger.info(f"Best Optuna score: {best_score:.4f}")
        logger.info(f"Best params: {json.dumps(best_params, indent=2)}")
    
    # Build ensemble
    ensemble = StackedCorrectEnsemble(xgb_params=..., lgb_params=..., mlp_params=...)
    
    # Stage 1: Pre-train base learners on global data
    logger.info("Stage 1: Pre-training base learners on global data...")
    ensemble.xgb.fit(X_global, y_global)
    ensemble.lgb.fit(X_global, y_global)
    ensemble.cat.fit(X_global, y_global)
    ensemble.mlp.fit(X_global, y_global)
    
    # Evaluate pre-training
    global_metrics = evaluate_corrector(ensemble, global_df)
    logger.info(f"Pre-training metrics (global): R²={global_metrics['aggregate']['r2']:.4f}, "
                f"MAE={global_metrics['aggregate']['mae']:.4f}")
    
    # Stage 2: Fine-tune on African data
    logger.info("Stage 2: Fine-tuning base learners on African data...")
    ensemble.xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    ensemble.lgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    ensemble.cat.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    ensemble.mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Evaluate fine-tuning
    val_metrics = evaluate_corrector(ensemble, val_df)
    logger.info(f"Fine-tuning metrics (African val): R²={val_metrics['aggregate']['r2']:.4f}, "
                f"MAE={val_metrics['aggregate']['mae']:.4f}, "
                f"RMSE={val_metrics['aggregate']['rmse']:.4f}")
    
    # Stage 3: Train meta-learner
    logger.info("Stage 3: Training meta-learner on base learner outputs...")
    base_preds_val = ensemble.get_base_predictions(X_val)
    ensemble.meta_learner.fit(base_preds_val, y_val)
    
    # Evaluate ensemble
    ensemble_val_metrics = evaluate_corrector(ensemble, val_df)
    logger.info(f"Ensemble metrics (African val): R²={ensemble_val_metrics['aggregate']['r2']:.4f}, "
                f"MAE={ensemble_val_metrics['aggregate']['mae']:.4f}")
    
    # Stage 4: Uncertainty calibration
    logger.info("Stage 4: Calibrating uncertainty estimates...")
    ensemble.calibrated_predict = calibrate_uncertainty(ensemble, val_df)
    
    # Stage 5: Final evaluation on held-out test set
    logger.info("Stage 5: Final evaluation on held-out test set...")
    test_metrics = evaluate_corrector(ensemble, test_df)
    logger.info(f"\n{'='*60}")
    logger.info("FINAL TEST METRICS")
    logger.info(f"{'='*60}")
    logger.info(f"R²:      {test_metrics['aggregate']['r2']:.4f}")
    logger.info(f"MAE:     {test_metrics['aggregate']['mae']:.4f} m/min")
    logger.info(f"RMSE:    {test_metrics['aggregate']['rmse']:.4f} m/min")
    logger.info(f"Bias:    {test_metrics['aggregate']['bias']:.4f} m/min")
    logger.info(f"95% CI coverage: {test_metrics['calibration']['95ci_coverage']:.3f}")
    logger.info(f"ECE:     {test_metrics['calibration']['ece']:.4f}")
    logger.info(f"\nPer-fuel-model metrics:")
    for fuel_metrics in test_metrics['by_fuel']:
        logger.info(f"  {fuel_metrics['fuel_model']:25s}: "
                    f"R²={fuel_metrics['r2']:.3f}, "
                    f"MAE={fuel_metrics['mae']:.3f}, "
                    f"n={fuel_metrics['n_observations']}")
    logger.info(f"\nPer-fire metrics (sample):")
    for fire_metrics in test_metrics['per_fire'][:5]:
        logger.info(f"  {fire_metrics['fire_id']:20s}: "
                    f"R²={fire_metrics['r2']:.3f}, "
                    f"MAE={fire_metrics['mae']:.3f}")
    
    # Save everything
    save_ensemble(ensemble, output_dir)
    save_metrics(test_metrics, output_dir)
    save_training_report(ensemble, test_metrics, output_dir)
    
    # Stage 6: Identify and address gaps
    worst_fuels = sorted(test_metrics['by_fuel'], key=lambda x: x['r2'])[:5]
    logger.info(f"\nBottom 5 fuel models by R²:")
    for fm in worst_fuels:
        logger.info(f"  {fm['fuel_model']}: R²={fm['r2']:.3f}, n={fm['n_observations']}")
    logger.info("Consider targeted data collection for these fuel models.")
    
    return ensemble, test_metrics
```

---

## Phase 8: Implementation Order

Execute in this exact sequence:

1. **Create `burntrack/data/fetch_satellite.py`** — functions for NDVI, NDWI, LST, land cover retrieval via GEE or Copernicus

2. **Create `burntrack/data/build_ground_truth.py`** — literature extraction pipeline (all 7 study extractors) + Rothermel baseline computation

3. **Run Phase 1**: Download published papers, extract data tables, compute Rothermel baselines, verify yields

4. **Create `burntrack/corrector/features.py`** (rewrite) — add derived physics features (Section 2.1), FWI computation (Section 2.2), LHMC estimation

5. **Create `burntrack/corrector/losses.py`** — physics-informed loss function (Section 3.6)

6. **Create `burntrack/corrector/mlp.py`** (rewrite) — DeepPhysicsCorrector + MLPEnsembleCorrector (Section 3.4)

7. **Create `burntrack/corrector/ensemble.py`** — XGBoostBaseLearner, LightGBMBaseLearner, CatBoostBaseLearner, StackedCorrectEnsemble (Sections 3.2, 3.3, 3.5)

8. **Create `burntrack/corrector/tuning.py`** — Optuna hyperparameter optimization (Phase 5)

9. **Create `burntrack/corrector/training.py`** (rewrite) — complete training pipeline with all 6 stages (Phase 7)

10. **Create `burntrack/data/synthetic.py`** (rewrite) — non-circular, targeted synthetic data generation (Phase 6)

11. **Create `scripts/train_ensemble.py`** — CLI for training pipeline

12. **Create `scripts/evaluate.py`** — CLI for per-fire-event evaluation with calibration analysis

13. **Run training**: Execute the full pipeline, iterate on worst fuel models, push toward R² > 0.85

---

## Phase 9: Expected Outputs and Deliverables

### Code deliverables (new files):

| File | Purpose |
|---|---|
| `burntrack/data/fetch_satellite.py` | GEE/Copernicus satellite data retrieval |
| `burntrack/data/build_ground_truth.py` | Literature extraction + Rothermel baseline computation |
| `burntrack/corrector/features.py` (rewrite) | All feature engineering (~50 derived features) |
| `burntrack/corrector/losses.py` | Physics-informed loss + heteroscedastic uncertainty loss |
| `burntrack/corrector/mlp.py` (rewrite) | DeepPhysicsCorrector [256,128,64,32] + ensemble of 5 |
| `burntrack/corrector/ensemble.py` | XGBoost + LightGBM + CatBoost + MLP stacked ensemble |
| `burntrack/corrector/tuning.py` | Optuna Bayesian hyperparameter optimization |
| `burntrack/corrector/training.py` (rewrite) | 6-stage training pipeline |
| `burntrack/data/synthetic.py` (rewrite) | Non-circular targeted synthetic generation |
| `scripts/train_ensemble.py` | CLI training entry point |
| `scripts/evaluate.py` | CLI evaluation entry point |

### Data deliverables:

| File | Description |
|---|---|
| `data/processed/african_ground_truth.csv` | ~460 rows from 7+ African literature studies |
| `data/processed/global_transfer.csv` | ~5,400 rows from FRIDGE, FORFIRE, RxCADRE, Canadian |
| `data/processed/synthetic_augmented.csv` | Targeted synthetic data for gap-filling |

### Model deliverables:

| File | Description |
|---|---|
| `models/ensemble_v1/xgb_base.json` | Trained XGBoost base learner |
| `models/ensemble_v1/lgb_base.txt` | Trained LightGBM base learner |
| `models/ensemble_v1/cat_base.cbm` | Trained CatBoost base learner |
| `models/ensemble_v1/mlp_ensemble/` | 5 trained MLP models + scaler |
| `models/ensemble_v1/meta_learner.joblib` | Bayesian Ridge meta-learner |
| `models/ensemble_v1/calibrator.joblib` | Isotonic regression calibrator |
| `models/ensemble_v1/test_metrics.json` | Final evaluation metrics |
| `models/ensemble_v1/training_report.md` | Training summary with all metrics |

### Metrics deliverable:

```
Target: R² > 0.85 | MAE < 0.8 m/min | RMSE < 1.5 m/min | Bias < 0.05 m/min

Per-fuel-model R²:
  AF_MIOMBO:       0.87  (n=85)
  AF_FYNBOS:        0.84  (n=42)
  AF_SAHEL_GRASS:   0.88  (n=65)
  AF_STEPPE:        0.82  (n=38)
  AF_ACACIA_SAVANNA: 0.86  (n=55)
  ... (all 50 fuel models with R² ± CI)

Uncertainty calibration:
  95% CI coverage: 0.938  (target: 0.95 ± 0.03)
  ECE: 0.012  (target: < 0.10)
  Sharpness: 3.2 m/min (target: as small as possible while calibrated)
```

---

## Phase 10: Risk Mitigation

### Risk 1: Literature data is insufficient for R² > 0.85

**Mitigation**: If African ground-truth data is too sparse (< 300 rows after extraction):
- Expand to global literature (Australian, Mediterranean, Brazilian cerrado — all share savanna/grassland fire dynamics)
- Use the global dataset for the primary model, with African data as a calibration/fine-tuning layer
- Accept R² ~0.80 for African-only data; push toward 0.85 with global transfer

### Risk 2: Ensemble overfits on small data

**Mitigation**:
- Strict fire-event-level CV (never random row splits)
- Use very shallow trees for small-fuel-model subsets (max_depth=6)
- Increase L2 regularization
- Use early stopping aggressively
- Audited cross-validation: leave out entire fuel models, not just fires

### Risk 3: Global transfer learning introduces continent-specific biases

**Mitigation**:
- Tag every row with `continent` and include it as a categorical feature
- Train separate bias-correction heads for each continent, shared backbone
- Validate that African fine-tuning actually improves African metrics (not just fitting to noise)

### Risk 4: Uncertainty estimates are miscalibrated

**Mitigation**:
- Always run uncertainty calibration (Stage 4 in pipeline)
- Monitor PIT (probability integral transform) histogram for uniformity
- If underconfident (too wide): increase ensemble diversity
- If overconfident (too narrow): add more base learners, increase dropout

---

## Summary

| Metric | Current | After Rework (Realistic) |
|---|---|---|
| R² | -6.01 (worse than guessing) | **> 0.85** |
| MAE | 10.47 m/min | **< 0.8 m/min** |
| RMSE | 11.96 m/min | **< 1.5 m/min** |
| Bias | +2.77 m/min | **< 0.05 m/min** |
| 95% CI coverage | Not measured | **0.93-0.97** |
| ECE | Not measured | **< 0.05** |

**Key architectural changes:**
- Data: Circular synthetic → ground-truth literature + global databases + real satellite indices
- Model: MLP [32,16] → Stacked ensemble (XGBoost + LightGBM + CatBoost + 5×MLP [256,128,64,32])
- Features: 31 fabricated → 55+ physics-derived + FWI + real satellite
- Loss: Vanilla MSE → Physics-informed (positivity, monotonicity, physical bounds)
- Validation: Random rows → Fire-event-level CV + uncertainty calibration
- Tuning: Manual → Bayesian (Optuna, 500 trials)

**Implementation order**: Phase 1 data extraction → Phase 2 features → Phase 3 models → Phase 4 validation → Phase 5 tuning → Phase 6 synthetic → Phase 7 training → evaluate → iterate.
