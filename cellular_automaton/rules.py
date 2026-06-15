# Module : Règles de propagation du feu
import numpy as np

UNBURNED = 0
BURNING = 1
ASH = 2

MOORE_OFFSETS = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1],
])

HEAT_OF_COMBUSTION = 18000.0
STEFAN_BOLTZMANN = 5.67e-8


# ──────────────────────────────────────────────
#  Formules physiques (cohérentes avec le dataset)
# ──────────────────────────────────────────────

def vpd_from_temp_rh(temp_c, rh_percent):
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return np.clip(es * (1.0 - rh_percent / 100.0), 0.0, None)


def dfmc_from_vpd_temp(vpd_kpa, temp_c):
    return np.clip(30.0 - 2.5 * vpd_kpa - 0.1 * temp_c, 0.0, 40.0)


def wind_midflame(wind_10m, fuel_bed_depth=0.6):
    if fuel_bed_depth < 0.6:
        return wind_10m * 0.4
    return wind_10m * np.clip(0.4 + 0.15 * (fuel_bed_depth - 0.6), 0.4, 1.0)


def phi_eff_vector(phi_w, phi_s, angle_wind_slope_deg):
    theta = np.deg2rad(angle_wind_slope_deg)
    return np.sqrt(phi_w**2 + phi_s**2 + 2.0 * phi_w * phi_s * np.cos(theta))


def rothermel_probability(ros_ms, cell_size=10.0, dt_s=60.0, p_nom=0.5):
    k = (ros_ms * dt_s) / cell_size
    p_base = 1.0 - (1.0 - p_nom) ** k
    return np.clip(p_base, 0.0, 1.0)


def moisture_factor(eta_M):
    return np.clip(eta_M, 0.0, 1.0)


def anisotropic_ros(ros_max, ros_back, wind_dir_deg, fire_dir_deg):
    theta = np.deg2rad(wind_dir_deg - fire_dir_deg)
    e = (ros_max - ros_back) / np.maximum(ros_max + ros_back, 1e-9)
    return ros_max * (1.0 - e) / np.maximum(1.0 - e * np.cos(theta), 1e-9)


def byram_intensity(ros_ms, fuel_consumed_kgm2):
    return HEAT_OF_COMBUSTION * fuel_consumed_kgm2 * ros_ms / 360.0


def radiative_preheat(burning_count, eta_M, T_flame=1200.0):
    preheat = STEFAN_BOLTZMANN * T_flame**4 * burning_count
    eta_M_reduction = np.clip(preheat * 5e-7, 0.0, eta_M * 0.5)
    return np.maximum(eta_M - eta_M_reduction, 0.0)


def spotting_distance(wind_speed_ms, flame_height=2.0):
    return wind_speed_ms * flame_height * 0.5


def fuel_pnom(sigma_m2_m3, delta_m, w_total_kg_m2, mx_percent):
    packing = np.clip(w_total_kg_m2 / np.maximum(delta_m, 0.01) / 513.0, 0.001, 1.0)
    efficiency = 1.0 - np.exp(-0.08 * sigma_m2_m3 * delta_m)
    dryness = np.clip(1.0 - mx_percent / 100.0, 0.1, 1.0)
    return np.clip(0.05 + 0.85 * efficiency * dryness * packing, 0.05, 0.95)


def corrected_ros(ros_rothermel, delta_ros):
    return np.maximum(ros_rothermel + delta_ros, 0.0)


# ──────────────────────────────────────────────
#  CellState
# ──────────────────────────────────────────────

class CellState:
    __slots__ = ('state', 'burn_time', 'fuel_load', 'fuel_initial',
                 'eta_M', 'eta_S', 'temperature', 'ros', 'phi_w', 'phi_s',
                 'wind_speed_ms', 'wind_dir_deg', 'slope_deg', 'aspect_deg',
                 'arrival_time', 'fireline_intensity', 'flame_height',
                 'beta', 'gamma', 'I_R_kW_m2', 'xi', 'tau_min')

    def __init__(self, shape, dtype=np.float32):
        self.state = np.zeros(shape, dtype=np.int8)
        self.burn_time = np.zeros(shape, dtype=np.int32)
        self.fuel_load = np.zeros(shape, dtype=dtype)
        self.fuel_initial = np.zeros(shape, dtype=dtype)
        self.eta_M = np.full(shape, 0.5, dtype=dtype)
        self.eta_S = np.full(shape, 0.95, dtype=dtype)
        self.temperature = np.full(shape, 300.0, dtype=dtype)
        self.ros = np.zeros(shape, dtype=dtype)
        self.phi_w = np.zeros(shape, dtype=dtype)
        self.phi_s = np.zeros(shape, dtype=dtype)
        self.wind_speed_ms = np.zeros(shape, dtype=dtype)
        self.wind_dir_deg = np.zeros(shape, dtype=dtype)
        self.slope_deg = np.zeros(shape, dtype=dtype)
        self.aspect_deg = np.zeros(shape, dtype=dtype)
        self.arrival_time = np.full(shape, -1, dtype=np.int32)
        self.fireline_intensity = np.zeros(shape, dtype=dtype)
        self.flame_height = np.zeros(shape, dtype=dtype)
        self.beta = np.zeros(shape, dtype=dtype)
        self.gamma = np.zeros(shape, dtype=dtype)
        self.I_R_kW_m2 = np.zeros(shape, dtype=dtype)
        self.xi = np.zeros(shape, dtype=dtype)
        self.tau_min = np.zeros(shape, dtype=dtype)


# ──────────────────────────────────────────────
#  FirePropagation
# ──────────────────────────────────────────────

class FirePropagation:
    def __init__(self, grid, cell_size=10.0, dt_s=60.0, p_nom=0.5,
                 wind_speed_ms=0.0, wind_dir_deg=0.0,
                 slope_deg=0.0, aspect_deg=0.0,
                 moisture_of_extinction=0.3, alpha=4.0,
                 ros_back_ratio=0.1, burn_duration=3,
                 fuel_load_kgm2=2.0, fuel_consumption_rate=0.3,
                 spotting_enabled=True, spotting_prob=0.02,
                 preheat_enabled=True, corrector_model=None,
                 random_seed=None):

        self.cell_size = cell_size
        self.dt_s = dt_s
        self.p_nom = p_nom
        self.ros_back_ratio = ros_back_ratio
        self.burn_duration = burn_duration
        self.fuel_consumption_rate = fuel_consumption_rate
        self.spotting_enabled = spotting_enabled
        self.spotting_prob = spotting_prob
        self.preheat_enabled = preheat_enabled
        self.corrector_model = corrector_model
        self.delta_ros_map = np.zeros_like(grid, dtype=np.float32)
        self.rng = np.random.default_rng(random_seed)
        self.rows, self.cols = grid.shape
        self.iteration = 0

        self.cs = CellState((self.rows, self.cols))
        self.cs.state[:] = grid
        self.cs.fuel_load[:] = fuel_load_kgm2
        self.cs.fuel_initial[:] = fuel_load_kgm2
        self.cs.wind_speed_ms[:] = wind_speed_ms
        self.cs.wind_dir_deg[:] = wind_dir_deg
        self.cs.slope_deg[:] = slope_deg
        self.cs.aspect_deg[:] = aspect_deg

    def set_rothermel_outputs(self, ros, phi_w, phi_s, beta, gamma,
                               I_R_kW_m2, xi, eta_M, eta_S, tau_min,
                               sigma_m2_m3=None, delta_m=None,
                               w_total_kg_m2=None, mx_percent=None):
        self.cs.ros = ros.astype(np.float32)
        self.cs.phi_w = phi_w.astype(np.float32)
        self.cs.phi_s = phi_s.astype(np.float32)
        self.cs.beta = beta.astype(np.float32)
        self.cs.gamma = gamma.astype(np.float32)
        self.cs.I_R_kW_m2 = I_R_kW_m2.astype(np.float32)
        self.cs.xi = xi.astype(np.float32)
        self.cs.eta_M = eta_M.astype(np.float32)
        self.cs.eta_S = eta_S.astype(np.float32)
        self.cs.tau_min = tau_min.astype(np.float32)
        if sigma_m2_m3 is not None:
            self.cs.p_nom_map = fuel_pnom(
                sigma_m2_m3,
                delta_m if delta_m is not None else 0.2,
                w_total_kg_m2 if w_total_kg_m2 is not None else 2.0,
                mx_percent if mx_percent is not None else 30.0,
            )

    def set_corrector(self, corrector_model):
        self.corrector_model = corrector_model
        self.delta_ros_map.fill(0.0)

    def apply_correction(self, features_dict):
        if self.corrector_model is None:
            return
        try:
            import pandas as pd
            df = pd.DataFrame([features_dict])
            raw = self.corrector_model.predict(df)[0]
            self.delta_ros_map[:, :] = raw if np.isscalar(raw) else raw.reshape(self.rows, self.cols)
        except Exception:
            pass

    def set_wind(self, speed_ms_map, dir_deg_map):
        self.cs.wind_speed_ms = speed_ms_map.astype(np.float32)
        self.cs.wind_dir_deg = dir_deg_map.astype(np.float32)

    def set_topography(self, slope_deg_map, aspect_deg_map):
        self.cs.slope_deg = slope_deg_map.astype(np.float32)
        self.cs.aspect_deg = aspect_deg_map.astype(np.float32)

    def set_fuel(self, fuel_map):
        self.cs.fuel_load = fuel_map.astype(np.float32)
        self.cs.fuel_initial = fuel_map.astype(np.float32)

    def set_eta_M(self, eta_M_map):
        self.cs.eta_M = eta_M_map.astype(np.float32)

    @property
    def state(self):
        return self.cs.state

    # ── helpers ─────────────────────────────

    def _neighbor_windows(self, arr):
        padded = np.pad(arr, ((1, 1), (1, 1)), mode='edge')
        return np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    def _burning_mask(self):
        return self.cs.state == BURNING

    def _unburned_mask(self):
        return self.cs.state == UNBURNED

    def _is_extinct(self):
        return not np.any(self.cs.state == BURNING)

    # ── sub-steps ────────────────────────────

    def _update_burning_cells(self):
        mask = self._burning_mask()
        self.cs.burn_time[mask] += 1

        tau = np.maximum(self.cs.tau_min, 1.0)
        tau_in_steps = np.ceil(tau * 60.0 / self.dt_s).astype(np.int32)
        self.cs.fuel_load[mask] -= (self.cs.fuel_initial[mask]
                                     / tau_in_steps[mask])
        fuel_exhausted = self.cs.fuel_load <= 0.0
        time_up = self.cs.burn_time >= tau_in_steps
        should_ash = mask & (time_up | fuel_exhausted)
        self.cs.state[should_ash] = ASH
        self.cs.fuel_load[should_ash] = 0.0

    def _preheat_step(self):
        if not self.preheat_enabled:
            return
        windows = self._neighbor_windows(self.cs.state.astype(np.int32))
        center = windows[:, :, 1, 1]
        burning_count = (np.sum(windows == BURNING, axis=(2, 3))
                         - (center == BURNING).astype(int))
        unburned = self._unburned_mask()
        if np.any(unburned):
            self.cs.eta_M[unburned] = radiative_preheat(
                burning_count[unburned], self.cs.eta_M[unburned]
            )

    def _propagation_step(self):
        new_state = self.cs.state.copy()
        new_arrival = self.cs.arrival_time.copy()
        new_intensity = self.cs.fireline_intensity.copy()
        new_flame_height = self.cs.flame_height.copy()

        burning = self._burning_mask()
        unburned = self._unburned_mask()

        if not np.any(burning) or not np.any(unburned):
            return new_state, new_arrival, new_intensity, new_flame_height

        p_max = np.zeros_like(self.cs.state, dtype=np.float32)
        ros_direction = np.zeros_like(self.cs.state, dtype=np.float32)

        for di, dj in MOORE_OFFSETS:
            ni = np.clip(np.arange(self.rows)[:, None] + di, 0, self.rows - 1)
            nj = np.clip(np.arange(self.cols)[None, :] + dj, 0, self.cols - 1)

            valid = ((np.arange(self.rows)[:, None] + di == ni) &
                     (np.arange(self.cols)[None, :] + dj == nj))

            eligible = burning[ni, nj] & valid & unburned
            if not np.any(eligible):
                continue

            fire_dir = np.rad2deg(np.arctan2(di, dj)) % 360
            ros_source = np.maximum(self.cs.ros[ni, nj], 0.01)
            ros_corrected = corrected_ros(
                ros_source, self.delta_ros_map[ni, nj]
            )
            ros_ij = anisotropic_ros(
                ros_corrected, ros_corrected * self.ros_back_ratio,
                self.cs.wind_dir_deg[ni, nj], fire_dir,
            )

            p_nom_cell = getattr(self.cs, 'p_nom_map',
                                 np.full_like(ros_ij, self.p_nom))
            p_base = rothermel_probability(
                ros_ij, self.cell_size, self.dt_s, p_nom_cell,
            )
            m = moisture_factor(self.cs.eta_M)
            p_final = np.clip(p_base * m, 0.0, 1.0)

            better = p_final > p_max
            p_max = np.maximum(p_max, p_final)
            ros_direction[better] = ros_ij[better]

        if not np.any(p_max > 0):
            return new_state, new_arrival, new_intensity, new_flame_height

        rand = self.rng.random(size=p_max.shape, dtype=np.float32)
        ignite = unburned & (rand < p_max)

        if np.any(ignite):
            new_state[ignite] = BURNING
            new_arrival[ignite] = np.where(
                ignite, self.iteration, new_arrival[ignite]
            )
            fuel_consumed = (self.cs.fuel_initial[ignite]
                             - self.cs.fuel_load[ignite] + 0.1)
            intensity = byram_intensity(ros_direction[ignite], fuel_consumed)
            new_intensity[ignite] = intensity
            new_flame_height[ignite] = np.clip(
                0.5 + intensity / 5000.0, 0.5, 15.0
            )
            self.cs.burn_time[ignite] = 0

        return new_state, new_arrival, new_intensity, new_flame_height

    def _spotting_step(self):
        if not self.spotting_enabled:
            return
        burning = self._burning_mask()
        high_intensity = self.cs.fireline_intensity > 3000.0
        sources = burning & high_intensity
        if not np.any(sources):
            return

        src_idx = np.argwhere(sources)
        chosen = src_idx[self.rng.integers(0, len(src_idx))]

        ws = self.cs.wind_speed_ms[chosen[0], chosen[1]]
        dist = int(spotting_distance(ws) / self.cell_size)
        if dist < 2:
            return
        rad = np.deg2rad(self.cs.wind_dir_deg[chosen[0], chosen[1]])
        ti = np.clip(chosen[0] + int(round(dist * np.sin(rad))), 0, self.rows - 1)
        tj = np.clip(chosen[1] + int(round(dist * np.cos(rad))), 0, self.cols - 1)

        if self.cs.state[ti, tj] == UNBURNED:
            if self.rng.random() < self.spotting_prob * moisture_factor(
                    self.cs.eta_M[ti, tj]):
                self.cs.state[ti, tj] = BURNING
                self.cs.burn_time[ti, tj] = 0
                self.cs.arrival_time[ti, tj] = self.iteration
                self.cs.fireline_intensity[ti, tj] = min(
                    byram_intensity(self.cs.ros[ti, tj],
                                    self.cs.fuel_initial[ti, tj]),
                    self.cs.fireline_intensity[chosen[0], chosen[1]] * 0.3,
                )

    # ── public API ──────────────────────────

    def step(self):
        self.iteration += 1
        self._update_burning_cells()
        if self._is_extinct():
            return False
        self._preheat_step()
        ns, na, ni, nh = self._propagation_step()
        self.cs.state[:] = ns
        self.cs.arrival_time[:] = na
        self.cs.fireline_intensity[:] = ni
        self.cs.flame_height[:] = nh
        self._spotting_step()
        return True

    def run(self, max_steps=1000):
        history = [self.cs.state.copy()]
        for _ in range(max_steps):
            if self._is_extinct():
                break
            self.step()
            history.append(self.cs.state.copy())
        return history

    def burned_area(self):
        return int(np.sum(self.cs.state == ASH))

    def active_fire_area(self):
        return int(np.sum(self.cs.state == BURNING))

    def probability_map(self):
        prob = np.zeros_like(self.cs.state, dtype=np.float32)
        prob[self.cs.state != UNBURNED] = 1.0
        return prob

    def summary(self):
        return {
            'iteration': self.iteration,
            'burning': self.active_fire_area(),
            'burned': self.burned_area(),
            'unburned': int(np.sum(self.cs.state == UNBURNED)),
            'max_intensity': float(self.cs.fireline_intensity.max()),
            'max_flame_height': float(self.cs.flame_height.max()),
        }


class EnsembleSimulation:
    def __init__(self, base_sim, n_realizations=100,
                 wind_dir_std=5.0, wind_speed_std=0.5,
                 eta_M_noise=0.03, spotting_noise=True):
        self.base_sim = base_sim
        self.n = n_realizations
        self.wind_dir_std = wind_dir_std
        self.wind_speed_std = wind_speed_std
        self.eta_M_noise = eta_M_noise
        self.spotting_noise = spotting_noise
        self.results = {}

    def run(self, max_steps=500):
        rows, cols = self.base_sim.rows, self.base_sim.cols
        prob_accum = np.zeros((rows, cols), dtype=np.float32)
        arrival_accum = np.zeros((rows, cols), dtype=np.float32)
        intensity_accum = np.zeros((rows, cols), dtype=np.float32)

        base = self.base_sim
        rng = np.random.default_rng(base.rng.integers(0, 2**31))

        for s in range(self.n):
            sim = FirePropagation(
                grid=base.state.copy(),
                cell_size=base.cell_size,
                dt_s=base.dt_s,
                p_nom=base.p_nom,
                ros_back_ratio=base.ros_back_ratio,
                burn_duration=base.burn_duration,
                fuel_load_kgm2=1.0,
                fuel_consumption_rate=base.fuel_consumption_rate,
                spotting_enabled=base.spotting_enabled,
                spotting_prob=base.spotting_prob,
                preheat_enabled=base.preheat_enabled,
                random_seed=s * 7919,
            )
            sim.set_rothermel_outputs(
                ros=np.maximum(base.cs.ros * (1.0 + rng.normal(0, 0.1, (rows, cols))), 0.0),
                phi_w=base.cs.phi_w,
                phi_s=base.cs.phi_s,
                beta=base.cs.beta,
                gamma=base.cs.gamma,
                I_R_kW_m2=base.cs.I_R_kW_m2,
                xi=base.cs.xi,
                eta_M=np.clip(base.cs.eta_M + rng.normal(0, self.eta_M_noise, (rows, cols)), 0.0, 1.0),
                eta_S=base.cs.eta_S,
                tau_min=base.cs.tau_min,
            )
            wind_rot = np.deg2rad(rng.normal(0, self.wind_dir_std))
            scale = 1.0 + rng.normal(0, self.wind_speed_std)
            sim.set_wind(
                base.cs.wind_speed_ms * scale,
                base.cs.wind_dir_deg + np.rad2deg(wind_rot),
            )
            sim.set_fuel(np.maximum(
                base.cs.fuel_initial * (1.0 + rng.normal(0, 0.05, (rows, cols))), 0.0
            ))
            sim.run(max_steps)

            prob_accum += sim.probability_map()
            with np.errstate(invalid='ignore'):
                arrival = np.where(sim.cs.arrival_time >= 0, sim.cs.arrival_time, max_steps)
                arrival_accum += arrival
                intensity_accum += np.where(sim.cs.fireline_intensity > 0, sim.cs.fireline_intensity, 0.0)

        self.results = {
            'probability': prob_accum / self.n,
            'mean_arrival': arrival_accum / self.n,
            'mean_intensity': intensity_accum / self.n,
            'n_realizations': self.n,
        }
        return self.results