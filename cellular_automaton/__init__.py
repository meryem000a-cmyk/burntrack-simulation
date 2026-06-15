"""
cellular_automaton — Module d'Automate Cellulaire BurnTrack
===========================================================
Simulation de propagation du feu couplée au modèle de Rothermel.

Composants :
    grid.py        : Grille et cellules (états, propriétés environnementales)
    rules.py       : Règles de propagation anisotropes via RothermelEngine
    simulation.py  : Runner, statistiques, point d'entrée principal

Usage rapide :
    from cellular_automaton import Grid, FireSimulation

    grid = Grid.uniform(50, 50, fuel_code="GR4", wind_speed_ms=5.0)
    sim = FireSimulation(grid, seed=42)
    sim.ignite(25, 25)
    sim.run(steps=90, dt=1.0, verbose=True)
"""

from .grid import Grid, Cell, CellState
from .rules import PropagationRules
from .simulation import FireSimulation, SimulationStats

__all__ = [
    "Grid",
    "Cell",
    "CellState",
    "PropagationRules",
    "FireSimulation",
    "SimulationStats",
]
