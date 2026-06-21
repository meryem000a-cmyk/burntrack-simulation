"""
cellular_automaton/simulation.py
=================================
Runner principal de la simulation de propagation du feu — BurnTrack.

Usage minimal :
    from cellular_automaton.grid import Grid
    from cellular_automaton.simulation import FireSimulation

    grid = Grid.uniform(50, 50, cell_size=30.0, fuel_code="GR4",
                        moisture_1h=0.06, wind_speed_ms=5.0, wind_dir_deg=270.0)
    sim = FireSimulation(grid, seed=42)
    sim.ignite(25, 25)
    stats = sim.run(steps=60, dt=1.0, verbose=True)   # 60 min de simulation
    print(f"Zone brûlée : {stats.summary()['final_burned_pct']:.1f}%")
"""

import time
import numpy as np
from typing import Callable, List, Optional, Tuple

from .grid import Grid, CellState
from .rules import PropagationRules


class SimulationStats:
    """Historique des statistiques collectées à chaque pas de temps."""

    def __init__(self):
        self.time_min: List[float] = []
        self.burning_cells: List[int] = []
        self.burned_fraction: List[float] = []
        self.new_ignitions: List[int] = []

    def record(self, t: float, burning: int, fraction: float, new: int):
        self.time_min.append(t)
        self.burning_cells.append(burning)
        self.burned_fraction.append(fraction)
        self.new_ignitions.append(new)

    def summary(self) -> dict:
        if not self.time_min:
            return {}
        return {
            "duration_min": self.time_min[-1],
            "peak_burning_cells": max(self.burning_cells),
            "final_burned_pct": self.burned_fraction[-1] * 100.0,
            "total_ignitions": sum(self.new_ignitions),
            "steps": len(self.time_min),
        }


class FireSimulation:
    """
    Simulation de propagation du feu par automate cellulaire couplé à Rothermel.

    Args:
        grid  : Grille initialisée (Grid)
        rules : Règles de propagation (PropagationRules). Défaut si None.
        seed  : Graine aléatoire pour reproductibilité (mode stochastique).
    """

    def __init__(
        self,
        grid: Grid,
        rules: Optional[PropagationRules] = None,
        seed: Optional[int] = None,
    ):
        self.grid = grid
        self.rules = rules or PropagationRules()
        self.current_time: float = 0.0   # minutes depuis le début
        self.step_count: int = 0
        self.stats = SimulationStats()
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Ignition
    # ------------------------------------------------------------------

    def ignite(self, row: int, col: int):
        """Allume manuellement une cellule (point de départ du feu)."""
        c = self.grid.cells[row][col]
        if c.state == CellState.UNBURNED:
            c.state = CellState.BURNING
            c.ignition_time = self.current_time
            c.burn_elapsed = 0.0
            # Durée de combustion via le temps de résidence Rothermel (tau)
            out = self.rules.compute_cell_ros(c, self.grid)
            tau = getattr(out, 'tau', None) or getattr(out, 'residence_time', None)
            c.burn_duration = max(10.0, float(tau)) if tau else 15.0

    def ignite_multiple(self, points: List[Tuple[int, int]]):
        """Allume plusieurs cellules simultanément."""
        for r, c in points:
            self.ignite(r, c)

    # ------------------------------------------------------------------
    # Pas de temps
    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> int:
        """
        Avance la simulation d'un pas de `dt` minutes.

        Returns:
            Nombre de nouvelles ignitions ce pas.
        """
        new_ignitions = self.rules.apply_step(self.grid, dt)
        n_new = len(new_ignitions) if isinstance(new_ignitions, list) else int(new_ignitions)
        self.current_time += dt
        self.step_count += 1
        self.stats.record(
            t=self.current_time,
            burning=self.grid.burning_count(),
            fraction=self.grid.burned_fraction(),
            new=n_new,
        )
        return n_new

    # ------------------------------------------------------------------
    # Runner complet
    # ------------------------------------------------------------------

    def run(
        self,
        steps: int,
        dt: float = 1.0,
        callback: Optional[Callable[["FireSimulation", int], None]] = None,
        stop_if_extinct: bool = True,
        verbose: bool = False,
    ) -> SimulationStats:
        """
        Lance la simulation pour `steps` pas de temps.

        Args:
            steps           : Nombre de pas de temps à simuler
            dt              : Durée d'un pas (minutes)
            callback        : Appelée après chaque pas → callback(sim, step_index)
                              Utile pour visualisation en temps réel ou logging externe.
            stop_if_extinct : Arrête si plus aucune cellule ne brûle.
            verbose         : Affiche la progression en console.

        Returns:
            SimulationStats avec l'historique complet.
        """
        t_wall = time.time()

        for s in range(steps):
            n_new = self.step(dt)

            if verbose and (s % max(1, steps // 10) == 0 or s == steps - 1):
                elapsed = time.time() - t_wall
                print(
                    f"[t={self.current_time:6.1f} min | {s+1:4d}/{steps}] "
                    f"burning={self.grid.burning_count():5d}  "
                    f"burned={self.grid.burned_fraction()*100:5.1f}%  "
                    f"+{n_new} new  ({elapsed:.1f}s réel)"
                )

            if callback:
                callback(self, s)

            if stop_if_extinct and self.grid.burning_count() == 0:
                if verbose:
                    print(f"\n[t={self.current_time:.1f} min] Feu éteint — fin de simulation.")
                break

        if verbose:
            s = self.stats.summary()
            print(f"\n=== Résumé simulation ===")
            for k, v in s.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

        return self.stats

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def snapshot(self) -> np.ndarray:
        """Copie de la grille d'états courants (int8 numpy array)."""
        return self.grid.state_array().copy()

    def reset(self):
        """Remet toutes les cellules à UNBURNED et réinitialise les compteurs."""
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                c = self.grid.cells[i][j]
                if c.state != CellState.FIREBREAK:
                    c.state = CellState.UNBURNED
        