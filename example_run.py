"""
Minimal working example of Crawl Space Optimisation (CSO).

Runs out of the box on synthetic data (no dataset download needed):

    python example_run.py

To use your own pre-processed SER features, replace `load_data()` with a
function returning (X, y), where y holds normalised valence/arousal in [0, 1].
Settings follow the paper: rate of change r = 0.1 (10%), restart after
m = 5 non-improving attempts, 80:20 train/test split with random_state=42.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from cso import HyperParameter, crawl_space_optimise, run_sweep

# Search space from Table II of the paper (initial range, global bounds).
SPACE = {
    "epochs":       HyperParameter("epochs", (5, 25), (5, 150), integer=True),
    "neurons":      HyperParameter("neurons", (10, 50), (10, 200), integer=True),
    "lr":           HyperParameter("lr", (0.0001, 0.2), (0.00001, 0.2)),
    "max_depth":    HyperParameter("max_depth", (3, 10), (1, 15), integer=True),
    "n_estimators": HyperParameter("n_estimators", (100, 800), (50, 1000), integer=True),
    "hidden":       HyperParameter("hidden", (1, 5), (1, 10), integer=True),
}

# Which hyper-parameters each model tunes.
MODEL_PARAMS = {
    "RF": ["max_depth", "n_estimators"],
    # Add neural models here, e.g. "MLP": ["lr", "epochs", "neurons", "hidden"]
}


def load_data():
    """Synthetic stand-in: 400 samples, 2 targets scaled to [0, 1]."""
    X, y = make_regression(n_samples=400, n_features=40, n_targets=2,
                           noise=20.0, random_state=0)
    y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
    return X, y


def make_rf_objective(X_train, X_test, y_train, y_test):
    def objective(point):
        model = RandomForestRegressor(
            max_depth=point["max_depth"],
            n_estimators=point["n_estimators"],
            n_jobs=-1,
            random_state=0,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return {
            "mae": mean_absolute_error(y_test, pred),  # aggregate over V and A
            "r2": r2_score(y_test, pred),
        }
    return objective


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 1) Single CSO run: Random Forest, 50 evaluations (as in Table III).
    result = crawl_space_optimise(
        make_rf_objective(X_train, X_test, y_train, y_test),
        [SPACE[p] for p in MODEL_PARAMS["RF"]],
        metric="mae", minimise=True,
        iterations=50, max_attempts=5, rate=0.1, seed=0,
    )
    restarts = result.history[-1].restart
    print(f"Best MAE {result.best_score:.4f} (R2 {result.best_metrics['r2']:.4f}) "
          f"at {result.best_point} after {len(result.history)} evaluations, "
          f"{restarts} restarts")

    # 2) Sweep: one CSO run per combination of fixed settings, all evaluations
    #    logged to CSV. In the paper the settings were model, feature
    #    extraction, audio visualisation and Keras optimiser.
    settings = {"model": ["RF"]}
    run_sweep(
        make_objective=lambda combo: make_rf_objective(X_train, X_test, y_train, y_test),
        settings=settings,
        space_for=lambda combo: [SPACE[p] for p in MODEL_PARAMS[combo["model"]]],
        output_csv="cso_sweep_results.csv",
        metric="mae", minimise=True, iterations=20, max_attempts=5, rate=0.1, seed=0,
    )
