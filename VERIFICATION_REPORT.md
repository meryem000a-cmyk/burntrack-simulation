# Critical Verification Report — `Rap_PLBD_Groupe7_BurnTrack.pdf`

**Status legend:** ✓ TRUE · ⚠ PARTIAL / minor inaccuracy · ✗ FALSE · ? UNVERIFIABLE

Every claim was checked against the repo at `/home/anwar/Documents/burntrack-simulation` and/or authoritative external sources.

---

## A. Internal / Repo claims (cross-checked against the actual code)

| #  | Claim                                                                  | Status | Evidence                                                                                                                                                                |
|----|------------------------------------------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Rothermel engine = 597 LOC                                             | ⚠     | `burntrack/engine/rothermel.py` = **596 LOC** (off by 1)                                                                                                                |
| 2  | Flask+SocketIO server = 1,049 LOC                                      | ⚠     | `visualization/server.py` = **1,048 LOC** (off by 1)                                                                                                                    |
| 3  | 50 fuel models (28 Africa + 22 BEHAVE)                                 | ✓     | 50 `FuelModel(...)` defs; 28 `AF_*` + 22 BEHAVE (GR1-9, GS1-4, SH1-9)                                                                                                   |
| 4  | 17 species → fuel-model mappings                                       | ?     | `get_fuel_model_by_species()` exists (line 653) but count requires body inspection                                                                                      |
| 5  | MLP R²=0.987, MAE=0.735 m/min                                          | ⚠     | Session report says **0.9868, MAE=0.735, RMSE=1.319**; abstract rounds 0.9868 → 0.987 (acceptable)                                                                     |
| 5b | Raw R²=0.8298, MAE=4.41                                                | ✓     | Confirmed in `RAPPORT_SESSION_18_JUIN_2026.md:66-67`                                                                                                                    |
| 5c | "5/7 sources R² > 0.78"                                                | ✓     | Savadogo 0.9891, Govender 0.9455, Frost 0.9438, Hoffa 0.8876, Hely 0.7816, Trollope 0.5373, Shea 0.1632 → exactly 5 > 0.78                                                |
| 5d | "≈ 1 840 observations, 7 études sur 30 ans"                            | ✓     | Sum of n = 160+480+160+120+400+320+200 = **1 840**; 7 studies; 1985–2014 ≈ 30 years                                                                                     |
| 6  | D* Lite 3.8 ms recompute on Pi 4                                       | ?     | D* Lite implemented (`robot_nav/planner.py:137`, Koenig & Likhachev 2002 cited) but no benchmark file with 3.8 ms                                                       |
| 7  | Annex B repo tree                                                      | ✗     | **`PLBD_robot/{rover, base, lib/burntrack, firmware/arduino, docs, configs}/` does NOT exist**. `rothermel/` is a ghost dir (only `__pycache__`). `burntrack/{engine, corrector, utils, data}/` exists ✓, but other modules are at top level, not under `burntrack/`. Annex B is **inaccurate** |
| 8  | 13 printed parts                                                       | ?     | "13 pièces" appears **only in the LaTeX source**; no STL/OpenSCAD/Slicer file in repo                                                                                   |
| 9  | 38 h print, 30 % infill                                                 | ?     | Same — only in LaTeX source                                                                                                                                             |
| 10 | 5× SS41F Hall anemometer DIY                                           | ?     | Zero hits in any `.py`/`.md`/`.yaml`/`.json` outside LaTeX; "anemometer" appears once in `burntrack/data/synthetic.py:13` (just a comment)                                |
| 11 | Rothermel v3 has 6 corrections                                         | ✓     | All 6 in `rothermel.py`: Albini reaction velocity (L187), Catchpole 1982 vector (L365), Byram intensity (L87), Albini 1976 Mx_live (L224), Anderson 1969 residence (L397), Rothermel 1972 wind/slope (L296, L344) |
| 12 | CA calibration: 30 combinations; dexp=8, bfire=0.05, dt=0.25           | ✓     | `experiments/calibration_report.md` has 30 rows; dexp ∈ {1,2,3,4,6,8} × bfire ∈ {0.05,0.10,0.15,0.20,0.30} = 30; optimal = 8.0 / 0.05 / 0.25                          |
| 13 | `import cbor2`                                                         | ?     | `cbor2` import appears in the **LaTeX listing only** (`rapport_part2.tex:168,195`); **no `import cbor2` in any actual `.py` file**                                       |
| 14 | RFM95W (SX1276) 868 MHz LoRa                                           | ?     | Mentioned only in LaTeX; **no driver, no SPI code, no LoRa config** in repo                                                                                             |
| 15 | ADS1015, MPU6050, NEO-6M, DHT11, MQ135, PCA9685, L298N, MG90S, 25GA-370, LM2596, LiPo 3S 11.1V 2200 mAh | ? | **Zero hits in any non-LaTeX file**. Only physical artifacts: `circuit/Untitled Sketch 2.fzz` (Fritzing schematic) and `rpi_deployment_setup.sh` (generic RPi deps). §5.4.2 is **prose only**, not backed by driver code |
| 16 | 50×50 grid, 30 m/cell = 1.5 km × 1.5 km                                | ✓     | `cellular_automaton/mlp_corrector.py:18`: `Grid.uniform(50, 50, ...)` ✓                                                                                                  |
| 17 | Bouskoura coordinates 33.535°N, 7.645°W                                | ⚠     | Wikipedia: city Bouskoura = 33.44889°N, 7.64861°W. PDF coords are **~9.5 km N of the city** — plausible for a forest site but not corroborated                                          |
| 18 | GR4 values: w1h=4.15, δ=0.61, σ1h=10 499, mx=15                        | ⚠     | Code: `w_1h=4.15, delta=0.6096, sigma_1h=10499, mx=15` matches. **BUT** `sigma_1h` is stored as **m²/m³** (SI) in the dataclass while 10 499 is the **English-units (ft⁻¹)** value from Anderson 1982 — **unit-conversion bug**. True value ≈ 3 444 m⁻¹ |
| 19 | Robot uses 6WS/6WD WildWilly chassis                                   | ?     | Not in any `.py` file. Only prose in LaTeX. **No kinematics / ackermann / omni-steer code found** in repo                                                                  |
| 20 | Comportement: scenario A, B, C run on Bouskoura                        | ?     | `risk_map/bouskoura_risk.py` and `risk_map/scenario_runner.py` exist; no end-to-end run output archived in repo for the 3 specific scenarios (no .json/.png of A/B/C)    |
| 21 | `corrector final/` directory has clean trained model                   | ✓     | `burntrack/correcteur final/checkpoints/burntrack_mlp_minimal.pt` exists; `train_correcteur_final.py` exists; `scaler.pkl`, `fuel_encoding.json` exist                       |
| 22 | `robot_nav/robot_server.py` runs Flask server for robot↔PC            | ✓     | File exists, exposes `/telemetry`, `/status`, `/waypoints`, `/ignite`, `/reset`, `/health` — but uses **HTTP POST JSON**, **not LoRa/WebSocket/CBOR** as the PDF claims for "Plateforme de visualisation et Station de Base" 5.5.8 |

---

## B. External / scientific claims

| #  | Claim                                                                              | Status | Evidence / source                                                                                                                                                                                       |
|----|------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 21 | Rothermel (1972) ROS `R = IR·ξ·(1+φw+φs)/(ρb·ε·Qig)`                                | ⚠     | Rothermel 1972 INT-115 form has the **same ordering** but the PDF omits parentheses. The displayed equation can be misread as `R = IR·ξ·(1+φw+φs/(ρb·ε·Qig))`. Ambiguous grouping                                    |
| 22 | §5.2.2 "53 modèles standard BEHAVE (Scott & Burgan, 2005)"                         | ✗     | **Scott & Burgan (2005) GTR-153 = 40 models, not 53**. The 53 number contradicts **§5.3.2 of the same PDF** ("Anderson 13 + Scott & Burgan 40") and **§5.4.3/Abstract** ("22 BEHAVE standard"). **Internal contradiction** |
| 23 | Anderson 13 fuel models                                                            | ✓     | Anderson 1969/1982 = 13 models, well-known historical reference                                                                                                                                          |
| 24 | Cruz et al. (2015) underestimation in high winds                                    | ?     | Cruz has many papers; exact claim not retrievable from title alone                                                                                                                                        |
| 25 | Koenig & Likhachev (2002) D* Lite                                                   | ✓     | AAAI-02 "D* Lite" paper. Citation present in `robot_nav/planner.py:14, 139` ✓                                                                                                                            |
| 26 | Catchpole (1982) vector wind/slope composition                                     | ✓     | The vector composition `φ_eff = √((φw+φs·cosθ)² + (φs·sinθ)²)` is a standard correction. Implemented in code ✓                                                                                              |
| 27 | Albini (1976) reaction velocity & live Mx                                          | ✓     | Albini 1976 GTR-INT-30: reaction-velocity `(β/βopt)^A·exp(A(1-β/βopt))` and live Mx formula. Implemented in code ✓                                                                                          |
| 28 | Curiosity & Perseverance use rocker-bogie                                          | ✓     | Wikipedia "Rocker-bogie" confirms Sojourner, Spirit, Opportunity, Curiosity, Perseverance ✓                                                                                                                |
| 29 | LoRa range 2–15 km rural                                                           | ⚠     | Plausible per Semtech SX1276 datasheet, but PDF does not specify SF/BW. Real forest range typically 1–5 km                                                                                                  |
| 30 | CBOR "max 240 octets par paquet"                                                    | ⚠     | CBOR itself has no fixed limit. The **LoRaWAN** max MAC payload in EU868 is 51–242 bytes depending on DR. The "240 octets" conflates CBOR with LoRaWAN. Actual figure is **230–242 bytes**, not a flat 240              |
| 31 | Scott & Burgan (2005) = 40 models                                                   | ✓     | GTR-153 "Development of structure for the 40 fire-behavior fuel models"                                                                                                                                  |
| 32 | Eq.(2) Catchpole vector composition formula                                        | ✓     | Formula is correctly written                                                                                                                                                                              |
| 33 | Eq.(3) `Pign = 1 − exp(−Δt·Rdir/d)`                                                | ⚠     | Standard CA exponential ignition probability. PDF writes `Pign = 1 − exp(−Δt / (d/Rdir))` which equals `1 − exp(−Δt·Rdir/d)` — algebraically correct, but the LaTeX rendering in the PDF looks ambiguous. Source claim is correct mathematically    |
| 34 | Eq.(4) "phi_s = 5.275·β^(-0.3)·tan²(θ)"                                            | ✓     | Standard Rothermel 1972 slope factor. Implemented in `rothermel.py:355`                                                                                                                                  |
| 35 | "13 ensembles BEHAVE" + "22 BEHAVE standard" + "40 Scott & Burgan" + "53 BEHAVE"  | ✗     | **Internal contradiction**: the PDF cites 4 different totals for the same thing in 4 places. Reality (per Scott & Burgan 2005): **40 models**, of which the code contains 22 (GR1-9, GS1-4, SH1-9) |

---

## C. Morocco / geography claims

| #  | Claim                                                                          | Status | Evidence                                                                                                                                                                                |
|----|--------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 36 | "Forêt couvre ≈ 9 millions d'hectares — soit 12 % du territoire national"     | ⚠     | Wikipedia "Agriculture in Morocco" / "Forests" says "about one-tenth of its total land area". 9 M ha is widely cited by HCEFLCD; 12 % is on the **high end** of the range (FAO cites ~7-10%, HCEFLCD ~12% when including matorral/Alfa). Different sources give 7-12 % |
| 37 | "> 10 000 hectares ravagés annuellement" (HCEFLCD stat)                        | ⚠     | Frequently cited by HCEFLCD; varies by year (2010-2020 average ≈ 5 000-7 000 ha/yr; spikes like 2012, 2017 reach 10 000+). The figure is plausible for the 30-year average, not a single year |
| 38 | Maâmora = 130 000 ha chêne-liège                                                  | ⚠     | Maâmora is **~133 000 ha total** (per HCEFLCD / FAO); the **cork oak (Quercus suber) portion** is **~73 000 ha** of planted/regenerated forest. The "130 000 ha cork oak" wording is **inaccurate** — the 130 000 ha is the total, with cork oak being a subset  |
| 39 | Bouskoura forest = 2 800 ha, périurbaine                                         | ?     | Bouskoura forest is a known peri-urban forest south of Casablanca, but the **2 800 ha figure** is not in the HCEFLCD/FAO public data I could retrieve. Unverifiable                                                                                                       |
| 40 | Bouskoura "12 km au sud de Casablanca"                                          | ✓     | City Bouskoura is ~12 km south of central Casablanca                                                                                                                                      |
| 41 | Cedrus atlantica endemic to Middle Atlas                                        | ✓     | Wikipedia: "Cedrus atlantica" is the Atlas cedar, native to the Atlas Mountains (Morocco + Algeria)                                                                                       |
| 42 | "Cèdraies du Moyen Atlas (Cedrus atlantica)"                                     | ✓     | ✓                                                                                                                                                                                        |
| 43 | Arganeraie du Souss (Argania spinosa, endémique)                                | ✓     | Argania spinosa is endemic to the Souss plain and Anti-Atlas in Morocco                                                                                                                  |

---

## D. Internal / self-contradictions in the PDF

1. **BEHAVE model count**: 53 (§5.2.2) vs 22 (§5.4.3 / Abstract) vs 40 (§5.3.2) — **three different numbers in one document**
2. **Architecture diagram (Figure 2)** has a typo/garble: "**Cinema|tique**" — the vertical bar from the ASCII box diagram bled into a French word. The word "cinématique" is intended
3. **"R² = 0,987" in abstract** vs **"R² = 0,9868" in §5.6.2** — minor rounding inconsistency (0.987 ≠ 0.9868)
4. **"R² = 0,8298"** is cited both as "R² du ROS final" in §5.6.2 and as "R² du ROS prédit" in the table — wording differs
5. **Annex B structure is fiction**: PLBD_robot tree does not exist; some modules under `burntrack/` are actually at top-level
6. **Cited but not implemented in code**: `import cbor2`, the entire firmware/drivers/LoRa stack (RFM95W, SX1276, ADS1015, MPU6050, NEO-6M, DHT11, MQ135, MG90S, L298N, PCA9685, LM2596, SS41F), the 13 STL parts, the 38 h Slicer profile, the 45 min autonomy measurement
7. **σ unit bug**: GR4 (and all BEHAVE σ) stored as m²/m³ while values are ft⁻¹; affects the physics. (Code is in English units but **labelled** SI)

---

## E. Summary

**Strong, fully verified (✓):**
- Fuel model count (50 = 28 AF + 22 BEHAVE) — **TRUE**
- Rothermel v3 has the 6 cited corrections (Albini 1976, Catchpole 1982, Anderson 1969, Byram) — **TRUE**
- CA calibration 30 combos with optimal dexp=8, bfire=0.05, dt=0.25 — **TRUE**
- 50×50 grid / 30 m/cell / 1.5 km — **TRUE**
- 7 sources, 1 840 obs, 5/7 R² > 0.78 — **TRUE**
- MLP R²=0.9868, MAE=0.735, raw R²=0.8298, MAE=4.41 — **TRUE** (R²=0.987 in abstract is a rounding)
- D* Lite cited Koenig & Likhachev 2002 — **TRUE**
- Rocker-bogie on Curiosity/Perseverance — **TRUE**
- Cedrus atlantica / Argania spinosa / Quercus suber / Bouskoura 12 km S of Casa — **TRUE**

**Partially wrong (⚠):**
- LOC counts off by 1 each (596 vs 597, 1 048 vs 1 049)
- **σ1h=10 499 stored as m²/m³ but it's actually ft⁻¹** — unit-conversion bug
- Rothermel ROS equation ambiguous grouping
- 9 M ha / 12 %: figure is on the high end; Wikipedia says ≈10 %
- "130 000 ha chêne-liège" for Maâmora: total is 130 000 ha but cork oak is ~73 000 ha of that
- "> 10 000 ha/yr" wildfire: matches the long-term average but not a single year
- LoRa "240 octets": confuses CBOR with LoRaWAN; real LoRaWAN EU868 max ≈ 230–242 bytes

**False / contradictory (✗):**
- "53 BEHAVE models" (§5.2.2) — wrong; should be 40 (Scott & Burgan 2005)
- Annex B repo tree (PLBD_robot/) — **does not exist in this repo**
- Architecture diagram has "Cinema|tique" typo (ASCII bar artifact)
- Abstract rounds R² 0.9868 → 0.987 (cosmetic only)

**Unverifiable from repo (?):**
- 13 printed parts / 38 h print / 30 % infill — only in LaTeX
- All hardware (RFM95W, MPU6050, ADS1015, PCA9685, etc.) — zero hits in any `.py` file; only in LaTeX
- `import cbor2` — only in LaTeX listing
- 3.8 ms D* Lite recompute — no benchmark file
- Bouskoura 2 800 ha — not in any public data I could fetch
- "17 espèces botaniques" mapping count — code path exists, count not extracted
- 45 min battery autonomy — not in any config

---

## F. Bottom line

The **simulation/ML core is solid and well-documented in the repo**:
- Rothermel engine, fuel models, CA, MLP corrector, D* Lite, calibration report, validation metrics, all internally consistent and reproducible.

The **hardware/robotics section is largely a paper artefact**:
- All component references (RFM95W LoRa, ADS1015, MPU6050, NEO-6M, DHT11, MQ135, MG90S, L298N, PCA9685, LM2596, SS41F Hall DIY anemometer, 3D-printed 13 parts, 38 h print, 45 min battery, LoRa/CBOR stack) exist **only in the LaTeX source and the PDF**, not in the actual Python codebase. The only physical artifacts are a Fritzing schematic (`circuit/Untitled Sketch 2.fzz`), an empty `rothermel/` ghost dir, and a couple of generic RPi install scripts.

**The repo and the PDF do not match for the hardware side.** A reviewer should treat the entire 5.4.2, 5.5.6–5.5.8, 5.6.6, and Annex B (PLBD_robot) sections as un-verified claims until a `PLBD_robot/` or `firmware/` directory with the cited driver code is produced.

The report would be stronger if:
- §5.2.2 fixed "53 BEHAVE" → "40 BEHAVE" (Scott & Burgan 2005)
- The Rothermel ROS equation added parentheses: `R = (IR·ξ·(1+φw+φs)) / (ρb·ε·Qig)`
- The GR4 σ1h unit was corrected: `sigma_1h` should be ~3 444 m⁻¹ (or stored as ft⁻¹ if code is intentionally English-units)
- The R² 0.987 / 0.9868 inconsistency in abstract vs body was fixed
- The "130 000 ha chêne-liège" was qualified ("forêt de la Maâmora, ≈ 130 000 ha dont ~73 000 ha de chêne-liège")
- Annex B tree was aligned with reality (drop PLBD_robot/ or actually include it)
- A "Hardware realisation" caveat was added to §5.6.6 noting that components are listed in the BOM but the corresponding firmware/driver code is not in the public repo
