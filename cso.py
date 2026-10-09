"""
Crawl Space Optimisation (CSO)

Reference implementation of the hyper-parameter optimiser described in:

    A. Dunmore, K. Gupta, E. Wen, A. Nassani, R. Wang, and M. Billinghurst,
    "Choices, Choices, Choices: Features, Datasets, and Optimisation for Speech
    Emotion Recognition," IEEE Transactions on Affective Computing, 2026,
    doi: 10.1109/TAFFC.2026.3733654.

CSO is a single-solution, iterated-local-search-style optimiser:

1. Sample a random start point from the initial ranges.
2. Perturb every hyper-parameter proportionally: x <- x + x * u, u ~ U(-r, +r),
   then round integer hyper-parameters and clip to the global bounds.
3. If the perturbed point improves on the current centre, it becomes the new
   centre (and the attempt counter resets).
4. After m consecutive attempts without improvement, restart from a new random
   start point.
5. The best point seen across all restarts is returned.

Each call to the objective counts as one iteration, so `iterations` is the
total evaluation budget (e.g. 50 per run in the paper's optimiser comparison).
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Number = float
Point = Dict[str, Number]
Objective = Callable[[Point], Mapping[str, Number]]


@dataclass
class HyperParameter:
    """One tunable hyper-parameter.

    start: (low, high) range used to draw random start/restart points.
    bounds: (low, high) global bounds; every candidate is clipped to these.
    integer: round to the nearest integer after perturbation.
    """

    name: str
    start: Tuple[Number, Number]
    bounds: Tuple[Number, Number]
    integer: bool = False

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if lo > hi:
            raise ValueError(f"{self.name}: lower bound {lo} > upper bound {hi}")
        s_lo, s_hi = self.start
        if s_lo > s_hi:
            raise ValueError(f"{self.name}: start range {self.start} is reversed")

    def clip(self, value: Number) -> Number:
        lo, hi = self.bounds
        value = min(max(value, lo), hi)
        if self.integer:
            value = int(round(value))
            value = min(max(value, int(math.ceil(lo))), int(math.floor(hi)))
        return value

    def sample_start(self, rng: random.Random) -> Number:
        lo, hi = self.start
        if self.integer:
            return self.clip(rng.randint(int(math.ceil(lo)), int(math.floor(hi))))
        return self.clip(rng.uniform(lo, hi))

    def perturb(self, value: Number, rate: float, rng: random.Random) -> Number:
        u = rng.uniform(-rate, rate)
        return self.clip(value + value * u)


@dataclass
class Evaluation:
    iteration: int
    restart: int
    point: Point
    metrics: Dict[str, Number]
    score: Number
    outcome: str  # "start", "improved", "no improvement"


@dataclass
class CSOResult:
    best_point: Point
    best_score: Number
    best_metrics: Dict[str, Number]
    history: List[Evaluation] = field(default_factory=list)


def crawl_space_optimise(
    objective: Objective,
    space: Sequence[HyperParameter],
    *,
    metric: str = "mae",
    minimise: bool = True,
    iterations: int = 100,
    max_attempts: int = 5,
    rate: float = 0.1,
    seed: Optional[int] = None,
    on_evaluation: Optional[Callable[[Evaluation], None]] = None,
) -> CSOResult:
    """Run CSO.

    objective: called with a dict {name: value}; must return a mapping that
        contains `metric` (other metrics are recorded but not optimised).
    metric / minimise: what to optimise and in which direction
        (MAE: minimise=True; R^2: minimise=False).
    iterations: total number of objective evaluations (i in Algorithm 1).
    max_attempts: consecutive non-improving attempts before a restart (m).
    rate: maximum proportional rate of change (r), e.g. 0.1 for 10%.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0 < max_attempts < iterations:
        raise ValueError("require 0 < max_attempts < iterations")
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    names = [hp.name for hp in space]
    if len(set(names)) != len(names):
        raise ValueError("hyper-parameter names must be unique")

    rng = random.Random(seed)
    sign = 1.0 if minimise else -1.0  # internally always minimise

    def evaluate(point: Point) -> Tuple[Dict[str, Number], Number]:
        metrics = dict(objective(dict(point)))
        if metric not in metrics:
            raise KeyError(
                f"objective returned {sorted(metrics)}, missing '{metric}'"
            )
        value = float(metrics[metric])
        if math.isnan(value):
            value = math.inf  # treat failed/NaN runs as worst possible
            return metrics, value
        return metrics, sign * value

    def random_point() -> Point:
        return {hp.name: hp.sample_start(rng) for hp in space}

    history: List[Evaluation] = []
    best_point: Optional[Point] = None
    best_score = math.inf
    best_metrics: Dict[str, Number] = {}

    restart = 0
    centre: Optional[Point] = None
    centre_score = math.inf
    attempts = 0

    for it in range(iterations):
        if centre is None:
            candidate = random_point()
            outcome_if_better = "start"
        else:
            candidate = {
                hp.name: hp.perturb(centre[hp.name], rate, rng) for hp in space
            }
            outcome_if_better = "improved"

        metrics, score = evaluate(candidate)

        if centre is None:
            centre, centre_score, attempts = candidate, score, 0
            outcome = "start"
        elif score < centre_score:
            centre, centre_score, attempts = candidate, score, 0
            outcome = outcome_if_better
        else:
            attempts += 1
            outcome = "no improvement"

        if score < best_score:
            best_point, best_score, best_metrics = dict(candidate), score, metrics

        record = Evaluation(it, restart, dict(candidate), metrics, sign * score, outcome)
        history.append(record)
        if on_evaluation is not None:
            on_evaluation(record)

        if attempts >= max_attempts:
            centre, centre_score, attempts = None, math.inf, 0
            restart += 1

    assert best_point is not None
    return CSOResult(best_point, sign * best_score, best_metrics, history)


# ---------------------------------------------------------------------------
# Sweep over fixed experimental settings (model, features, visualisation, ...)
# ---------------------------------------------------------------------------

def run_sweep(
    make_objective: Callable[[Dict[str, object]], Objective],
    settings: Mapping[str, Sequence[object]],
    space_for: Callable[[Dict[str, object]], Sequence[HyperParameter]],
    *,
    output_csv: str,
    metric: str = "mae",
    minimise: bool = True,
    iterations: int = 100,
    max_attempts: int = 5,
    rate: float = 0.1,
    seed: Optional[int] = None,
) -> List[Tuple[Dict[str, object], Optional[CSOResult]]]:
    """Run CSO once for every combination of `settings`.

    make_objective(combo) returns the objective for that combination.
    space_for(combo) returns the hyper-parameters to tune for that combination
    (e.g. max_depth/n_estimators for RF; lr/epochs/neurons for neural models).
    Every evaluation is written to `output_csv` (rewritten after each combination). A failing combination is
    reported on stderr and the sweep continues; nothing is silently swallowed.
    """
    keys = list(settings)
    combos = [dict(zip(keys, values)) for values in itertools.product(*settings.values())]
    results: List[Tuple[Dict[str, object], Optional[CSOResult]]] = []

    rows: List[Dict[str, object]] = []

    def write_csv() -> None:
        fieldnames: List[str] = []
        for row in rows:
            fieldnames += [k for k in row if k not in fieldnames]
        with open(output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)

    for idx, combo in enumerate(combos):
        combo_seed = None if seed is None else seed + idx

        def log(ev: Evaluation, combo=combo) -> None:
            rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                         **{f"setting_{k}": v for k, v in combo.items()},
                         "iteration": ev.iteration, "restart": ev.restart,
                         "outcome": ev.outcome,
                         **{f"hp_{k}": v for k, v in ev.point.items()},
                         **{f"metric_{k}": v for k, v in ev.metrics.items()}})

        try:
            res = crawl_space_optimise(
                make_objective(combo), list(space_for(combo)), metric=metric,
                minimise=minimise, iterations=iterations, max_attempts=max_attempts,
                rate=rate, seed=combo_seed, on_evaluation=log,
            )
            print(f"[{idx + 1}/{len(combos)}] {combo}: best {metric} = "
                  f"{res.best_score:.4f} at {res.best_point}")
            results.append((combo, res))
        except Exception:
            print(f"[{idx + 1}/{len(combos)}] {combo}: FAILED", file=sys.stderr)
            traceback.print_exc()
            results.append((combo, None))
        write_csv()  # rewrite after every combination so partial results survive

    return results
