"""Polynomial regression with fold-local preprocessing and nested model selection.

The inner search shares transformed arrays only between candidates in the SAME
degree/fold. Warm starts share coefficients only along one regularization path
on the SAME training rows. Nothing is shared across validation boundaries.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Candidate:
    degree: int
    model: str
    alpha: float
    l1_ratio: float | None = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic checkpoint: an interrupted write cannot masquerade as complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config, problem):
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("Both fold counts must be at least two.")
    cap = {1: 10, 2: 20}[problem]
    degree = config["max_degrees"][str(problem)]
    if not isinstance(degree, int) or not 1 <= degree <= cap:
        raise ValueError(f"Problem {problem} requires a maximum degree in 1..{cap}.")
    for key in ("ridge_alphas", "sparse_alphas"):
        if not config[key] or any(not np.isfinite(a) or a <= 0 for a in config[key]):
            raise ValueError(f"{key} must contain finite positive values.")
    if not config["l1_ratios"] or any(not 0 < r < 1 for r in config["l1_ratios"]):
        raise ValueError("Elastic Net l1_ratios must be strictly between zero and one.")
    if config["max_iter"] < 1 or not 0 < config["tol"] < 1:
        raise ValueError("Invalid solver iteration limit or tolerance.")


def read_features(path, features, target=False):
    frame = pd.read_csv(path)
    expected = features + (["y"] if target else [])
    if list(frame.columns) != expected:
        raise ValueError(f"{path}: expected columns {expected}, got {list(frame.columns)}")
    values = frame.to_numpy(dtype=np.float64)
    if not len(frame) or not np.isfinite(values).all():
        raise ValueError(f"{path}: data must be nonempty, numeric, and finite.")
    return frame


def make_preprocessor(degree):
    return Pipeline([
        ("input_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("term_scaler", StandardScaler()),
    ])


def make_estimator(candidate, config, warm_start=False):
    if candidate.model == "ridge":
        # Cholesky automatically uses a dual solve when features exceed samples.
        return Ridge(alpha=candidate.alpha, solver="cholesky")
    options = dict(alpha=candidate.alpha, max_iter=config["max_iter"],
                   tol=config["tol"], selection="random", random_state=config["seed"],
                   warm_start=warm_start)
    if candidate.model == "lasso":
        return Lasso(**options)
    if candidate.model == "elasticnet":
        return ElasticNet(l1_ratio=candidate.l1_ratio, **options)
    raise ValueError(f"Unknown model: {candidate.model}")


def make_pipeline(candidate, config):
    return Pipeline(make_preprocessor(candidate.degree).steps + [
        ("regressor", make_estimator(candidate, config))
    ])


def candidates_for_degree(degree, config):
    result = [Candidate(degree, "ridge", float(a)) for a in config["ridge_alphas"]]
    # Descending alpha supports warm starts from stronger to weaker penalties.
    alphas = sorted(config["sparse_alphas"], reverse=True)
    result += [Candidate(degree, "lasso", float(a)) for a in alphas]
    for ratio in config["l1_ratios"]:
        result += [Candidate(degree, "elasticnet", float(a), float(ratio)) for a in alphas]
    return result


def evaluate_degree(degree, X, y, splits, config, checkpoint):
    """Inner CV for one degree; preprocessors are fitted anew in each fold."""
    if checkpoint is not None and Path(checkpoint).exists():
        return read_json(checkpoint)
    candidates = candidates_for_degree(degree, config)
    records = [dict(candidate=asdict(c), fold_mse=[], fold_r2=[], failures=[]) for c in candidates]
    with threadpool_limits(limits=1):
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            transform = make_preprocessor(degree)
            train = np.asfortranarray(transform.fit_transform(X[train_idx]))
            valid = transform.transform(X[valid_idx])
            estimators = {}
            for candidate, record in zip(candidates, records):
                path_key = (candidate.model, candidate.l1_ratio)
                try:
                    estimator = estimators.get(path_key)
                    if estimator is None or candidate.model == "ridge":
                        estimator = make_estimator(candidate, config, warm_start=True)
                        estimators[path_key] = estimator
                    estimator.set_params(alpha=candidate.alpha)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(train, y[train_idx])
                    convergence = [str(w.message) for w in caught
                                   if issubclass(w.category, ConvergenceWarning)]
                    if convergence:
                        raise RuntimeError("Solver did not converge: " + convergence[0])
                    prediction = estimator.predict(valid)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Non-finite predictions")
                    record["fold_mse"].append(float(mean_squared_error(y[valid_idx], prediction)))
                    record["fold_r2"].append(float(r2_score(y[valid_idx], prediction)))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    record["fold_mse"].append(None)
                    record["fold_r2"].append(None)
                    record["failures"].append(dict(fold=fold, reason=str(error)))
                    estimators.pop(path_key, None)
    for record in records:
        valid = not record["failures"]
        record["mean_mse"] = float(np.mean(record["fold_mse"])) if valid else None
        record["std_mse"] = float(np.std(record["fold_mse"], ddof=1)) if valid else None
        record["mean_r2"] = float(np.mean(record["fold_r2"])) if valid else None
    if checkpoint is not None:
        write_json(checkpoint, records)
    best = min((r["mean_mse"] for r in records if r["mean_mse"] is not None), default=None)
    failed = sum(bool(r["failures"]) for r in records)
    print(f"  degree {degree:2d}: best inner MSE={best}, invalid candidates={failed}", flush=True)
    return records


def search(X, y, config, problem, jobs=1, checkpoint_dir=None):
    splits = list(KFold(n_splits=config["inner_folds"], shuffle=True,
                        random_state=config["seed"]).split(X))
    maximum = config["max_degrees"][str(problem)]
    results = Parallel(n_jobs=jobs)(delayed(evaluate_degree)(
        degree, X, y, splits, config,
        None if checkpoint_dir is None else Path(checkpoint_dir) / f"degree_{degree:02d}.json"
    ) for degree in range(1, maximum + 1))
    records = [record for degree_records in results for record in degree_records]
    eligible = [r for r in records if r["mean_mse"] is not None]
    if not eligible:
        raise RuntimeError("No candidate converged on all inner folds. Inspect checkpoints.")
    winner = min(eligible, key=lambda r: (r["mean_mse"], r["candidate"]["degree"]))
    return Candidate(**winner["candidate"]), records


def fit_selected(candidate, X, y, config):
    pipeline = make_pipeline(candidate, config)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y)
    return pipeline


def nested_evaluate(X, y, config, problem, jobs=1, checkpoint_dir=None):
    folds = KFold(n_splits=config["outer_folds"], shuffle=True,
                  random_state=config["seed"] + 1)
    oof = np.full(len(y), np.nan)
    assignments = np.full(len(y), -1, dtype=int)
    scores = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X), start=1):
        print(f"Outer fold {fold}/{config['outer_folds']}", flush=True)
        root = None if checkpoint_dir is None else Path(checkpoint_dir) / f"outer_{fold}"
        result_file = None if root is None else root / "evaluation.json"
        if result_file is not None and result_file.exists():
            result = read_json(result_file)
            prediction = np.asarray(result["predictions"])
            if result["validation_indices"] != valid_idx.tolist():
                raise ValueError("Checkpoint fold indices do not match.")
        else:
            selected, records = search(X[train_idx], y[train_idx], config, problem, jobs, root)
            pipeline = fit_selected(selected, X[train_idx], y[train_idx], config)
            prediction = pipeline.predict(X[valid_idx])
            best = min(r["mean_mse"] for r in records if r["mean_mse"] is not None)
            result = dict(fold=fold, selected=asdict(selected), inner_selection_mse=best,
                          mse=float(mean_squared_error(y[valid_idx], prediction)),
                          r2=float(r2_score(y[valid_idx], prediction)),
                          validation_indices=valid_idx.tolist(), predictions=prediction.tolist())
            if result_file is not None:
                write_json(result_file, result)
        oof[valid_idx] = prediction
        assignments[valid_idx] = fold
        scores.append({k: v for k, v in result.items()
                       if k not in ("validation_indices", "predictions")})
        print(f"  outer MSE={result['mse']:.6g}, R2={result['r2']:.6g}; {result['selected']}", flush=True)
    summary = dict(
        outer_folds=scores,
        mean_outer_mse=float(np.mean([s["mse"] for s in scores])),
        std_outer_mse=float(np.std([s["mse"] for s in scores], ddof=1)),
        mean_outer_r2=float(np.mean([s["r2"] for s in scores])),
        std_outer_r2=float(np.std([s["r2"] for s in scores], ddof=1)),
        pooled_oof_mse=float(mean_squared_error(y, oof)),
        pooled_oof_r2=float(r2_score(y, oof)),
    )
    return summary, oof, assignments


def write_predictions(path, predictions):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": predictions}).to_csv(path, index=False)
    return path


def train_problem(data_dir, rollno, problem, config, output, jobs=1):
    validate_config(config, problem)
    data_dir, output = Path(data_dir), Path(output)
    features = [f"x{i}" for i in range(1, {1: 6, 2: 3}[problem] + 1)]
    train_file = data_dir / f"{rollno}_train_var{problem}.csv"
    test_file = data_dir / f"{rollno}_test_var{problem}.csv"
    training = read_features(train_file, features, target=True)
    testing = read_features(test_file, features)
    X, y = training[features].to_numpy(), training["y"].to_numpy()
    minimum_train = len(y) - int(np.ceil(len(y) / config["outer_folds"]))
    if len(y) < 2 * config["outer_folds"] or minimum_train < 2 * config["inner_folds"]:
        raise ValueError("Too few rows to have at least two validation rows per fold.")
    root = output / f"var{problem}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=config, problem=problem, rollno=rollno,
                    train_sha256=file_hash(train_file), test_sha256=file_hash(test_file),
                    implementation_sha256=file_hash(__file__), features=features,
                    versions={"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "sklearn": sklearn.__version__})
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError(f"{root}: configuration, data, code, or versions changed. Use a new --output directory.")
    write_json(manifest_path, manifest)
    summary, oof, assignments = nested_evaluate(X, y, config, problem, jobs, root / "search")
    pd.DataFrame(dict(row=np.arange(len(y)), outer_fold=assignments, y=y, prediction=oof)).to_csv(
        root / "outer_predictions.csv", index=False)
    write_json(root / "nested_metrics.json", summary)
    print("Final selection on all training rows (outer scores are not used to select).", flush=True)
    selected, records = search(X, y, config, problem, jobs, root / "search" / "final")
    pipeline = fit_selected(selected, X, y, config)
    prediction = pipeline.predict(testing.to_numpy())
    model_path = root / "model.joblib"
    joblib.dump(dict(pipeline=pipeline, features=features, selected=asdict(selected),
                     manifest=manifest), model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored["pipeline"].predict(testing.to_numpy()), prediction)
    submission = write_predictions(output / "predictions" / f"{rollno}_pred_var{problem}.csv", prediction)
    n_terms = len(pipeline.named_steps["regressor"].coef_)
    nonzero = int(np.count_nonzero(pipeline.named_steps["regressor"].coef_))
    best_record = next(r for r in records if r["candidate"] == asdict(selected))
    selection = dict(selected=asdict(selected), final_selection_cv_mse=best_record["mean_mse"],
                     terms=n_terms, nonzero_coefficients=nonzero,
                     training_rows=len(training), test_rows=len(testing),
                     prediction_file=str(submission), model_file=str(model_path),
                     note="Final selection CV is not an independent evaluation; use nested_metrics.json.")
    write_json(root / "final_selection.json", selection)
    pd.DataFrame([dict(**r["candidate"], mean_mse=r["mean_mse"], std_mse=r["std_mse"],
                      mean_r2=r["mean_r2"], failed_folds=len(r["failures"]))
                  for r in records]).to_csv(root / "final_search.csv", index=False)
    print(json.dumps(dict(problem=problem, **selection, mean_outer_mse=summary["mean_outer_mse"]), indent=2), flush=True)
    return selection
