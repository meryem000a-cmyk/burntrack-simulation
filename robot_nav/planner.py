"""
robot_nav/planner.py
=====================
Navigation autonome du robot en terrain incendie -- BurnTrack.

Composants
----------
RiskMap
    Dijkstra multi-sources depuis les cellules BURNING.
    Produit arrival_time[(i,j)] = minutes avant que le feu atteigne (i,j).
    Utilise le ROS directionnel de Rothermel pour chaque arete.

DStarLite
    Planificateur de chemin dynamique (Koenig & Likhachev 2002).
    Planifie a rebours goal -> start dans une grille 8-directionnelle.
    Quand le feu evolue, seuls les sommets affectes sont remis a jour --
    bien plus efficace qu un A* replanifie depuis zero a chaque etape.

    Couts d arete :
        - BURNING / t_arrive < safety_margin  ->  INF (cellule bloquee)
        - t_arrive dans [margin, 2*margin]    ->  dist * risk_penalty (risque)
        - FIREBREAK (chemin, route, pare-feu) ->  dist * 0.5 (couloir prefere)
        - cellule sure                         ->  dist

RobotNavigator
    Interface haut niveau.
    Gere la position du robot, la re-planification periodique et l historique
    de trajectoire. Appeler .step() a chaque pas de simulation.

Usage minimal
-------------
    from cellular_automaton import Grid, FireSimulation, PropagationRules
    from robot_nav.planner import RobotNavigator

    grid  = Grid.uniform(50, 50, fuel_code="GR4")
    rules = PropagationRules()
    sim   = FireSimulation(grid, rules, seed=0)
    sim.ignite(0, 0)

    robot = RobotNavigator(position=(49, 0), goal=(5, 45), safety_margin_min=10.0)
    for _ in range(120):
        sim.step(dt=1.0)
        status = robot.step(grid, rules, sim.current_time)
        if status != "navigating":
            break
    print(robot.status, len(robot.path_history), "steps")
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellular_automaton.grid import Grid, CellState
from cellular_automaton.rules import PropagationRules, _to_roth_fuel

INF = float("inf")
Pos = Tuple[int, int]   # (row, col)


# ---------------------------------------------------------------------------
# RiskMap
# ---------------------------------------------------------------------------

class RiskMap:
    """
    Carte des temps d arrivee du feu (minutes) pour chaque cellule.

    Dijkstra multi-sources depuis toutes les cellules BURNING simultanement.
    Pour chaque arete (src -> voisin), le temps de traversee = dist / ROS(src, dir).
    """

    def __init__(self):
        self.arrival_time: Dict[Pos, float] = {}

    def build(self, grid: Grid, rules: PropagationRules) -> Dict[Pos, float]:
        """
        (Re)construit la carte depuis l etat courant de la grille.

        Returns:
            dict {(row,col): float}  --  INF = cellule hors portee / ne brule pas.
        """
        arrival: Dict[Pos, float] = {
            (i, j): (0.0 if grid.cells[i][j].state == CellState.BURNING else INF)
            for i in range(grid.rows)
            for j in range(grid.cols)
        }

        heap = [
            (0.0, i, j)
            for (i, j), t in arrival.items()
            if t == 0.0
        ]
        heapq.heapify(heap)
        visited: Set[Pos] = set()

        while heap:
            t_src, i, j = heapq.heappop(heap)
            if (i, j) in visited:
                continue
            visited.add((i, j))

            src = grid.cells[i][j]
            if src.state in (CellState.BURNED, CellState.FIREBREAK):
                continue

            fm_raw = grid.get_fuel(src.fuel_code)
            if fm_raw is None:
                continue
            fuel = _to_roth_fuel(fm_raw)

            for ni, nj, dist, spread_dir in grid.neighbors(i, j):
                if (ni, nj) in visited:
                    continue
                tgt = grid.cells[ni][nj]
                if tgt.state in (CellState.BURNED, CellState.FIREBREAK):
                    continue

                ros = rules.compute_cell_ros(src, fuel, spread_dir)
                if ros < rules.min_ros:
                    continue

                t_arrive = t_src + dist / ros
                if t_arrive < arrival.get((ni, nj), INF):
                    arrival[(ni, nj)] = t_arrive
                    heapq.heappush(heap, (t_arrive, ni, nj))

        self.arrival_time = arrival
        return arrival


# ---------------------------------------------------------------------------
# DStarLite
# ---------------------------------------------------------------------------

class DStarLite:
    """
    D* Lite -- planificateur de chemin dynamique (Koenig & Likhachev 2002).

    Notation originale de l article :
        g[s]   = distance estimee de s au goal
        rhs[s] = valeur lookahead (min sur successeurs : cost(s,s') + g[s'])
        U      = file de priorite des sommets inconsistants
        k_m    = correction cumulee des cles apres deplacement du robot
    """

    def __init__(self, grid: Grid,
                 safety_margin_min: float = 10.0,
                 risk_penalty: float = 4.0):
        self.grid          = grid
        self.safety_margin = safety_margin_min
        self.risk_penalty  = risk_penalty

        self.g:    Dict[Pos, float] = {}
        self.rhs:  Dict[Pos, float] = {}
        self.U:    List             = []
        self.U_set: Set[Pos]        = set()
        self.k_m:  float            = 0.0

        self.start:       Optional[Pos] = None
        self.goal:        Optional[Pos] = None
        self.last_start:  Optional[Pos] = None
        self.arrival_time: Dict[Pos, float] = {}
        self.current_time: float            = 0.0

    # ------------------------------------------------------------------
    # Cout d arete u -> v
    # ------------------------------------------------------------------

    def _edge_cost(self, u: Pos, v: Pos, dist: float) -> float:
        vi, vj = v
        cell = self.grid.cells[vi][vj]

        if cell.state in (CellState.BURNING, CellState.BURNED):
            return INF
        if cell.state == CellState.FIREBREAK:
            return dist * 0.5          # couloir sur -- prefere

        t_arrive  = self.arrival_time.get(v, INF)
        time_left = t_arrive - self.current_time

        if time_left < self.safety_margin:
            return INF                 # trop dangereux
        if time_left < 2.0 * self.safety_margin:
            return dist * self.risk_penalty   # zone a risque
        return dist

    # ------------------------------------------------------------------
    # Heuristique octile (admissible pour 8-directions)
    # ------------------------------------------------------------------

    def _h(self, s: Pos) -> float:
        dr = abs(s[0] - self.start[0])
        dc = abs(s[1] - self.start[1])
        cs = self.grid.cell_size
        return cs * (max(dr, dc) + (np.sqrt(2) - 1.0) * min(dr, dc))

    # ------------------------------------------------------------------
    # Cle de priorite
    # ------------------------------------------------------------------

    def _key(self, s: Pos) -> Tuple[float, float]:
        m = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (m + self._h(s) + self.k_m, m)

    # ------------------------------------------------------------------
    # File de priorite (min-heap avec lazy deletion)
    # ------------------------------------------------------------------

    def _push(self, s: Pos):
        heapq.heappush(self.U, (self._key(s), s))
        self.U_set.add(s)

    def _top_key(self) -> Optional[Tuple[float, float]]:
        while self.U:
            k, s = self.U[0]
            if s in self.U_set:
                return k
            heapq.heappop(self.U)
        return None

    def _pop(self) -> Optional[Pos]:
        while self.U:
            k, s = heapq.heappop(self.U)
            if s in self.U_set:
                self.U_set.discard(s)
                return s
        return None

    # ------------------------------------------------------------------
    # Mise a jour d un sommet (coeur de D* Lite)
    # ------------------------------------------------------------------

    def _update_vertex(self, u: Pos):
        if u != self.goal:
            best = INF
            for ni, nj, dist, _ in self.grid.neighbors(u[0], u[1]):
                c = self._edge_cost(u, (ni, nj), dist)
                v = c + self.g.get((ni, nj), INF)
                if v < best:
                    best = v
            self.rhs[u] = best

        self.U_set.discard(u)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)

    # ------------------------------------------------------------------
    # Calcul du chemin optimal (boucle principale D* Lite)
    # ------------------------------------------------------------------

    def _compute_shortest_path(self):
        while True:
            top = self._top_key()
            if top is None:
                break
            if not (top < self._key(self.start)
                    or self.rhs.get(self.start, INF) != self.g.get(self.start, INF)):
                break

            u = self._pop()
            if u is None:
                break

            k_new = self._key(u)
            if top < k_new:
                self._push(u)
            elif self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for ni, nj, dist, _ in self.grid.neighbors(u[0], u[1]):
                    self._update_vertex((ni, nj))
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for ni, nj, dist, _ in self.grid.neighbors(u[0], u[1]):
                    self._update_vertex((ni, nj))

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def initialize(self, start: Pos, goal: Pos,
                   arrival_time: Dict[Pos, float],
                   current_time: float = 0.0):
        """
        Initialise D* Lite. A appeler une seule fois au debut.

        Args:
            start        : Position initiale du robot (row, col)
            goal         : Objectif (row, col)
            arrival_time : Sortie de RiskMap.build()
            current_time : Temps courant (min)
        """
        self.start        = start
        self.goal         = goal
        self.last_start   = start
        self.arrival_time = arrival_time
        self.current_time = current_time
        self.k_m          = 0.0
        self.g            = {}
        self.rhs          = {goal: 0.0}
        self.U            = []
        self.U_set        = set()
        self._push(goal)
        self._compute_shortest_path()

    def replan(self, new_start: Pos,
               new_arrival_time: Dict[Pos, float],
               current_time: float):
        """
        Re-planifie apres deplacement du robot et evolution du feu.

        Seuls les sommets dont le temps d arrivee du feu a change
        sont mis a jour -- c est l avantage cle de D* Lite.

        Args:
            new_start        : Nouvelle position du robot
            new_arrival_time : Carte de risque mise a jour
            current_time     : Temps courant (min)
        """
        self.k_m         += self._h(self.last_start)
        self.last_start   = self.start
        self.start        = new_start
        self.current_time = current_time

        # Identifier les cellules dont le cout a change
        changed: Set[Pos] = set()
        for pos in set(self.arrival_time) | set(new_arrival_time):
            if abs(self.arrival_time.get(pos, INF) - new_arrival_time.get(pos, INF)) > 0.5:
                changed.add(pos)
        self.arrival_time = new_arrival_time

        for pos in changed:
            self._update_vertex(pos)
            for ni, nj, dist, _ in self.grid.neighbors(pos[0], pos[1]):
                self._update_vertex((ni, nj))

        self._compute_shortest_path()

    def next_move(self) -> Optional[Pos]:
        """
        Retourne la prochaine cellule vers laquelle se deplacer.
        None si le robot est piege (pas de chemin sur vers le goal).
        """
        if self.g.get(self.start, INF) >= INF:
            return None
        si, sj = self.start
        best, best_cost = None, INF
        for ni, nj, dist, _ in self.grid.neighbors(si, sj):
            s2  = (ni, nj)
            val = self._edge_cost(self.start, s2, dist) + self.g.get(s2, INF)
            if val < best_cost:
                best_cost = val
                best = s2
        return best

    def get_full_path(self) -> List[Pos]:
        """Extrait le chemin complet start -> goal pour visualisation."""
        path    = [self.start]
        current = self.start
        seen    = {current}
        for _ in range(self.grid.rows * self.grid.cols):
            if current == self.goal:
                break
            ci, cj = current
            best, best_cost = None, INF
            for ni, nj, dist, _ in self.grid.neighbors(ci, cj):
                s2 = (ni, nj)
                if s2 in seen:
                    continue
                val = self._edge_cost(current, s2, dist) + self.g.get(s2, INF)
                if val < best_cost:
                    best_cost = val
                    best = s2
            if best is None or best_cost >= INF:
                break
            path.append(best)
            seen.add(best)
            current = best
        return path


# ---------------------------------------------------------------------------
# RobotNavigator
# ---------------------------------------------------------------------------

class RobotNavigator:
    """
    Interface haut niveau : robot autonome naviguant dans la simulation de feu.

    Appeler .step(grid, rules, sim.current_time) a chaque pas de simulation.
    Le robot se deplace d une cellule par appel (vitesse = cell_size / dt m/min).
    Pour un rover lent, appeler .step() toutes les N etapes de simulation.

    Args:
        position          : Position initiale (row, col)
        goal              : Objectif a atteindre (row, col)
        safety_margin_min : Marge de securite (min). Cellules dont le feu
                            arrive dans < N min sont evitees.
        replan_every      : Frequence de re-planification (en appels a .step()).
    """

    def __init__(self, position: Pos, goal: Pos,
                 safety_margin_min: float = 10.0,
                 replan_every: int = 5):
        self.position       = position
        self.goal           = goal
        self.safety_margin  = safety_margin_min
        self.replan_every   = replan_every
        self.path_history: List[Pos] = [position]
        self.status: str             = "navigating"
        self._planner: Optional[DStarLite] = None
        self._risk_map = RiskMap()
        self._steps    = 0

    def step(self, grid: Grid, rules: PropagationRules,
             current_time: float) -> str:
        """
        Avance le robot d une cellule.

        Etapes :
          1. Initialisation D* Lite au premier appel.
          2. Re-planification si periodicite atteinte ou danger immediat.
          3. Recuperation du prochain mouvement et deplacement.

        Returns:
            "navigating" | "reached" | "trapped"
        """
        if self.status != "navigating":
            return self.status

        # Premier appel : initialiser
        if self._planner is None:
            arrival = self._risk_map.build(grid, rules)
            self._planner = DStarLite(grid, self.safety_margin)
            self._planner.initialize(self.position, self.goal, arrival, current_time)

        # Danger immediat sur la cellule courante ?
        t_here    = self._risk_map.arrival_time.get(self.position, INF)
        in_danger = (
            grid.cells[self.position[0]][self.position[1]].state == CellState.BURNING
            or (t_here - current_time) < self.safety_margin
        )

        if self._steps >= self.replan_every or in_danger:
            new_arrival = self._risk_map.build(grid, rules)
            self._planner.replan(self.position, new_arrival, current_time)
            self._steps = 0
        else:
            self._steps += 1

        # Prochain mouvement
        nxt = self._planner.next_move()
        if nxt is None:
            self.status = "trapped"
            return self.status

        self.position = nxt
        self.path_history.append(nxt)
        self._planner.start = nxt

        if self.position == self.goal:
            self.status = "reached"
        return self.status

    @property
    def planned_path(self) -> List[Pos]:
        """Chemin planifie courant start -> goal (liste de (row,col))."""
        if self._planner is None:
            return [self.position]
        return self._planner.get_full_path()
