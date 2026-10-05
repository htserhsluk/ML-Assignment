"""
Polynomial Regression Assignment
--------------------------------
Implements:
  * PolynomialFeatures
  * Ridge
  * Lasso
  * ElasticNet
  * 80/20 validation split for degree/model/alpha selection
  * threaded hyperparameter fitting with ThreadPoolExecutor
  * random candidate polynomial-term selection + F-statistic screening
  * final refit on all training data
  * prediction CSV generation for var1 and var2

Expected files:
  IMT2024065_train_var1.csv
  IMT2024065_test_var1.csv
  IMT2024065_train_var2.csv
  IMT2024065_test_var2.csv

Outputs:
  IMT2024065_pred_var1.csv
  IMT2024065_pred_var2.csv
"""

from __future__ import annotations

# IMPORTANT: keep BLAS single-threaded so our Python thread pool does not
# oversubscribe the CPU with another layer of native threads.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ------------------------------ CONFIG --------------------------------------

ROLL_NO = "IMT2024065"

FILES = {
    "var1": {
        "train": f"{ROLL_NO}_train_var1.csv",
        "test": f"{ROLL_NO}_test_var1.csv",
        "max_degree": 10,
    },
    "var2": {
        "train": f"{ROLL_NO}_train_var2.csv",
        "test": f"{ROLL_NO}_test_var2.csv",
        "max_degree": 20,
    },
}

# Hold-out validation is allowed by the assignment and is much faster than
# nested CV when testing many polynomial degrees.
VALIDATION_SIZE = 0.20
RANDOM_SEED = 2026

# Thread pool size. Put 0 to auto-use (CPU count - 1), with a sensible cap.
THREADS = 0

# Above this number of polynomial terms, use the random candidate +
# f_regression screening step. This is the main speed safeguard.
MAX_FEATURES = 800

# We preserve all linear and quadratic terms, then screen higher-degree terms.
PRESERVE_UP_TO_DEGREE = 2

# Randomly sample at most this many higher-degree terms before F-score ranking.
# A fixed seed makes the run reproducible.
RANDOM_CANDIDATES = 2500

# To save a lot of time, Ridge is used as the cheap first pass for every
# degree. Lasso/ElasticNet are then run only on the best few degrees.
TOP_DEGREES_FOR_L1 = 5

# Hyperparameters are generated from the actual design matrix instead of
# being tied to arbitrary fixed alpha values. The generated grids adapt to
# the scale and dimensionality of each polynomial problem.
ALPHA_GRID_SIZE = 12
L1_GRID_SIZE = 12
L1_RATIO_GRID_SIZE = 7

# Search/final convergence settings are derived from the number of samples and
# selected features, rather than fixed iteration counts.
SEARCH_MAX_ITER_FACTOR = 20
FINAL_MAX_ITER_FACTOR = 80
SEARCH_TOL = 2e-4
FINAL_TOL = 1e-5


# --------------------------- DATA STRUCTURES --------------------------------

@dataclass
class ModelResult:
    problem: str
    degree: int
    n_features: int
    model: str
    alpha: float
    l1_ratio: Optional[float]
    val_mse: float
    val_r2: float
    elapsed_sec: float


# ---------------------------- HELPER FUNCTIONS ------------------------------

def get_thread_count() -> int:
    if THREADS and THREADS > 0:
        return THREADS
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 1, 12))


def load_dataset(problem: str):
    cfg = FILES[problem]
    train = pd.read_csv(cfg["train"])
    test = pd.read_csv(cfg["test"])

    if "y" not in train.columns:
        raise ValueError(f"{cfg['train']} must contain a 'y' column.")

    feature_cols = [c for c in train.columns if c != "y"]
    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"{cfg['test']} is missing columns: {missing}")

    X = train[feature_cols].to_numpy(dtype=np.float64)
    y = train["y"].to_numpy(dtype=np.float64)
    X_test = test[feature_cols].to_numpy(dtype=np.float64)

    return X, y, X_test, feature_cols


def standardize_for_search(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def generate_ridge_alphas(X: np.ndarray, n_values: int = ALPHA_GRID_SIZE):
    """Generate Ridge alphas from the spectrum/scale of the current X.
    For standardized X, the mean diagonal of X^T X / n is near one. Using
    the largest eigenvalue gives a data-dependent upper scale, while a log
    grid explores weak through strong regularization.
    """
    Xs, _ = standardize_for_search(X)
    gram = (Xs.T @ Xs) / max(1, len(Xs))
    scale = float(np.linalg.eigvalsh(gram)[-1])
    scale = max(scale, 1e-8)

    # The span is generated from dimensionality rather than fixed alphas.
    lower = scale * (1.0 / max(10.0, X.shape[1] * 10.0))
    upper = scale * max(10.0, np.sqrt(len(X)) * 2.0)
    return np.unique(np.geomspace(lower, upper, n_values))


def alpha_max_standardized(X: np.ndarray, y: np.ndarray) -> float:
    """Maximum L1 alpha, calculated from the current data."""
    Xs, _ = standardize_for_search(X)
    amax = float(np.max(np.abs(Xs.T @ y)) / len(y))
    return max(amax, np.finfo(float).eps)


def generate_l1_alphas(X: np.ndarray, y: np.ndarray, n_values: int = L1_GRID_SIZE):
    """Generate Lasso/ElasticNet alpha values relative to alpha_max."""
    amax = alpha_max_standardized(X, y)
    # Lower bound adapts to sample count; with more samples we can afford a
    # finer search near weak regularization.
    lower_fraction = max(1e-5, 1.0 / (len(y) ** 0.5 * 20.0))
    return np.geomspace(amax * lower_fraction, amax, n_values)


def generate_l1_ratios(X: np.ndarray, n_values: int = L1_RATIO_GRID_SIZE):
    """Generate a sensible L1/L2 mixture grid from problem dimensionality."""
    p = X.shape[1]
    # More features -> include more L2-heavy candidates; fewer features ->
    # allow stronger sparsity. Endpoints are derived from p.
    l2_heavy = 1.0 / (1.0 + np.log1p(p))
    low = max(0.05, l2_heavy)
    high = min(0.98, 1.0 - l2_heavy / 2.0)
    return np.linspace(low, high, n_values)


def generated_max_iter(X: np.ndarray, factor: int) -> int:
    return max(5000, factor * max(X.shape[0], X.shape[1]))


def select_polynomial_terms(
    X_train_poly: np.ndarray,
    X_valid_poly: np.ndarray,
    y_train: np.ndarray,
    powers: np.ndarray,
    max_features: int,
    seed: int,
    random_candidates: int,
):
    """
    Fast polynomial-term screening.

    1. Keep all terms with total degree <= PRESERVE_UP_TO_DEGREE.
    2. Randomly sample a candidate pool from higher-degree terms.
    3. Rank the sampled candidates by univariate F-statistic against y.
    4. Keep the best terms until max_features is reached.

    This is substantially safer than pure random deletion because the random
    step limits computation while the F-screening step keeps informative terms.
    Selection is performed ONLY on the training split during validation.
    """
    n_terms = X_train_poly.shape[1]
    if n_terms <= max_features:
        return X_train_poly, X_valid_poly, np.arange(n_terms, dtype=np.int32)

    term_degrees = powers.sum(axis=1)
    keep = np.flatnonzero(term_degrees <= PRESERVE_UP_TO_DEGREE)
    candidates = np.flatnonzero(term_degrees > PRESERVE_UP_TO_DEGREE)

    rng = np.random.default_rng(seed)
    if candidates.size > random_candidates:
        candidates = rng.choice(candidates, size=random_candidates, replace=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores, _ = f_regression(X_train_poly[:, candidates], y_train)

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    room = max_features - keep.size
    if room <= 0:
        selected = keep[:max_features]
    else:
        take = min(room, len(candidates))
        # argpartition is much faster than sorting every candidate.
        local = np.argpartition(scores, -take)[-take:]
        top = candidates[local]
        selected = np.concatenate([keep, top])

    selected = np.unique(selected)
    selected.sort()

    return (
        X_train_poly[:, selected],
        X_valid_poly[:, selected],
        selected.astype(np.int32),
    )


def prepare_degree_data(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    degree: int,
    problem: str,
):
    """
    Build polynomial features exactly once for a degree, then optionally reduce
    the number of terms.
    """
    t0 = time.perf_counter()

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        order="C",
    )

    Xtr_poly = poly.fit_transform(X_train)
    Xva_poly = poly.transform(X_valid)

    selected_train, selected_valid, selected_idx = select_polynomial_terms(
        Xtr_poly,
        Xva_poly,
        y_train,
        poly.powers_,
        max_features=MAX_FEATURES,
        seed=RANDOM_SEED + 1000 * (1 if problem == "var1" else 2) + degree,
        random_candidates=RANDOM_CANDIDATES,
    )

    # Scaling is done once per degree and reused by all regularized models.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(selected_train)
    Xva = scaler.transform(selected_valid)

    elapsed = time.perf_counter() - t0
    return poly, scaler, Xtr, Xva, selected_idx, elapsed


def fit_and_score(spec, Xtr, Xva, ytr, yva, problem, degree):
    """
    Fit one model configuration. This function is intentionally independent,
    so ThreadPoolExecutor can execute many configurations concurrently.
    """
    model_name, alpha, l1_ratio = spec
    t0 = time.perf_counter()

    if model_name == "ridge":
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="auto",
        )
    elif model_name == "lasso":
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=generated_max_iter(Xtr, SEARCH_MAX_ITER_FACTOR),
            tol=SEARCH_TOL,
            selection="cyclic",
        )
    elif model_name == "elasticnet":
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=generated_max_iter(Xtr, SEARCH_MAX_ITER_FACTOR),
            tol=SEARCH_TOL,
            selection="cyclic",
        )
    else:
        raise ValueError(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    mse = float(mean_squared_error(yva, pred))
    r2 = float(r2_score(yva, pred))

    return ModelResult(
        problem=problem,
        degree=degree,
        n_features=Xtr.shape[1],
        model=model_name,
        alpha=float(alpha),
        l1_ratio=None if l1_ratio is None else float(l1_ratio),
        val_mse=mse,
        val_r2=r2,
        elapsed_sec=time.perf_counter() - t0,
    )


def threaded_grid_search(
    specs,
    Xtr,
    Xva,
    ytr,
    yva,
    problem,
    degree,
    max_workers,
):
    """Run model configurations in parallel using a thread pool."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="polyfit") as pool:
        futures = [
            pool.submit(
                fit_and_score,
                spec,
                Xtr,
                Xva,
                ytr,
                yva,
                problem,
                degree,
            )
            for spec in specs
        ]

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r.val_mse)
    return results


def ridge_specs(X):
    alphas = generate_ridge_alphas(X)
    return [("ridge", float(a), None) for a in alphas]


def l1_specs(y_train, Xtr_scaled):
    alphas = generate_l1_alphas(Xtr_scaled, y_train)
    ratios = generate_l1_ratios(Xtr_scaled)

    specs = [("lasso", float(a), None) for a in alphas]
    for a in alphas:
        for l1 in ratios:
            specs.append(("elasticnet", float(a), float(l1)))
    return specs


def fit_final_model(
    model_name: str,
    alpha: float,
    l1_ratio: Optional[float],
    X_full_scaled: np.ndarray,
    y: np.ndarray,
    X_test_scaled: np.ndarray,
):
    """Fit the selected model on all training data and predict the test set."""
    if model_name == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
    elif model_name == "lasso":
        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=generated_max_iter(X_full_scaled, FINAL_MAX_ITER_FACTOR),
            tol=FINAL_TOL,
            selection="cyclic",
        )
    elif model_name == "elasticnet":
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=generated_max_iter(X_full_scaled, FINAL_MAX_ITER_FACTOR),
            tol=FINAL_TOL,
            selection="cyclic",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_full_scaled, y)

    return model, model.predict(X_test_scaled)


# ---------------------------- PROBLEM SOLVER --------------------------------

def solve_problem(problem: str, max_workers: int):
    cfg = FILES[problem]
    X, y, X_test, feature_cols = load_dataset(problem)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    all_results: list[ModelResult] = []
    cached = {}

    print(f"\n{'=' * 78}")
    print(f"{problem.upper()} | train={len(X)} | features={len(feature_cols)}")
    print(f"Testing degrees 1..{cfg['max_degree']}")
    print(f"MAX_FEATURES={MAX_FEATURES}, threads={max_workers}")
    print(f"{'=' * 78}")

    # ---------------- First pass: Ridge for EVERY degree ----------------
    for degree in range(1, cfg["max_degree"] + 1):
        poly, scaler, Xtr, Xva, selected_idx, prep_time = prepare_degree_data(
            X_train, X_valid, y_train, degree, problem
        )

        ridge_candidates = ridge_specs(Xtr)
        ridge_results = threaded_grid_search(
            ridge_candidates,
            Xtr,
            Xva,
            y_train,
            y_valid,
            problem,
            degree,
            max_workers=max_workers,
        )

        best_ridge = ridge_results[0]
        all_results.extend(ridge_results)

        cached[degree] = {
            "poly": poly,
            "scaler": scaler,
            "Xtr": Xtr,
            "Xva": Xva,
            "selected_idx": selected_idx,
            "prep_time": prep_time,
            "ridge_best": best_ridge,
        }

        print(
            f"[Ridge] degree={degree:2d} "
            f"features={Xtr.shape[1]:4d} "
            f"val_MSE={best_ridge.val_mse:.6f} "
            f"R2={best_ridge.val_r2:.6f} "
            f"alpha={best_ridge.alpha:.6g} "
            f"tested={len(ridge_candidates)} "
            f"prep={prep_time:.2f}s"
        )

    # Keep the best few degrees from Ridge for the expensive L1 models.
    ranked_degrees = sorted(
        cached.keys(),
        key=lambda d: cached[d]["ridge_best"].val_mse
    )
    candidate_degrees = ranked_degrees[:TOP_DEGREES_FOR_L1]

    print(f"\nRunning Lasso + ElasticNet on degrees: {candidate_degrees}")

    # ---------------- Second pass: Lasso + ElasticNet ----------------
    for degree in candidate_degrees:
        item = cached[degree]
        specs = l1_specs(y_train, item["Xtr"])
        print(f"  degree={degree:2d}: testing {len(specs)} Lasso/ElasticNet configurations")

        l1_results = threaded_grid_search(
            specs,
            item["Xtr"],
            item["Xva"],
            y_train,
            y_valid,
            problem,
            degree,
            max_workers=max_workers,
        )
        all_results.extend(l1_results)

        best = l1_results[0]
        print(
            f"[L1 ] degree={degree:2d} "
            f"features={item['Xtr'].shape[1]:4d} "
            f"best={best.model:10s} "
            f"val_MSE={best.val_mse:.6f} "
            f"R2={best.val_r2:.6f} "
            f"alpha={best.alpha:.6g} "
            f"l1={best.l1_ratio}"
        )

    # Global winner using validation MSE.
    winner = min(all_results, key=lambda r: r.val_mse)

    print(f"\nBEST {problem.upper()}:")
    print(asdict(winner))

    # ---------------------- Final fit on ALL training data ----------------
    degree = winner.degree
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        order="C",
    )
    X_poly_full = poly.fit_transform(X)
    X_poly_test = poly.transform(X_test)

    X_full_sel, X_test_sel, selected_idx = select_polynomial_terms(
        X_poly_full,
        X_poly_test,
        y,
        poly.powers_,
        max_features=MAX_FEATURES,
        seed=RANDOM_SEED + 1000 * (1 if problem == "var1" else 2) + degree,
        random_candidates=RANDOM_CANDIDATES,
    )

    # Standardize after selection.
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full_sel)
    X_test_scaled = final_scaler.transform(X_test_sel)

    final_model, predictions = fit_final_model(
        model_name=winner.model,
        alpha=winner.alpha,
        l1_ratio=winner.l1_ratio,
        X_full_scaled=X_full_scaled,
        y=y,
        X_test_scaled=X_test_scaled,
    )

    output_path = f"{ROLL_NO}_pred_{problem}.csv"
    pd.DataFrame({"y": predictions}).to_csv(output_path, index=False)

    print(
        f"Saved {output_path} | test_rows={len(predictions)} | "
        f"selected_features={len(selected_idx)}"
    )

    return winner, all_results


# ---------------------------------- MAIN ------------------------------------

def main():
    # Fail early with a clear message if an expected CSV is missing.
    missing = []
    for problem, cfg in FILES.items():
        for kind, filename in cfg.items():
            if kind in ("train", "test") and not Path(filename).exists():
                missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Missing required CSV file(s):\\n  "
            + "\\n  ".join(missing)
            + "\\n\\nPlace the CSV files in the same folder as this script."
        )

    max_workers = get_thread_count()
    print(f"Using {max_workers} worker threads.")

    for problem in ("var1", "var2"):
        solve_problem(problem, max_workers)

    print("\nDone.")
    print(f"Generated: {ROLL_NO}_pred_var1.csv")
    print(f"Generated: {ROLL_NO}_pred_var2.csv")


if __name__ == "__main__":
    main()
