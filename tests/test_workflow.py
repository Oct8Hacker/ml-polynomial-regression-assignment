import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import polynomial_regression as pr


def small_config():
    return dict(seed=7, outer_folds=3, inner_folds=2, max_degrees={"1": 2, "2": 2},
                ridge_alphas=[0.01, 1.0], sparse_alphas=[0.01, 0.1],
                l1_ratios=[0.5], max_iter=20000, tol=1e-6)


def test_inner_preprocessing_matches_independent_fold_pipeline():
    # A strong shift makes fitting scalers before splitting detectably wrong.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    X[-10:] += 8
    y = 2 * X[:, 0] - X[:, 1] ** 2 + rng.normal(scale=0.1, size=50)
    config = small_config()
    splits = list(KFold(2, shuffle=True, random_state=7).split(X))
    records = pr.evaluate_degree(2, X, y, splits, config, None)
    record = next(r for r in records if r["candidate"]["model"] == "ridge")
    candidate = pr.Candidate(**record["candidate"])
    expected = []
    for train, valid in splits:
        pipeline = pr.fit_selected(candidate, X[train], y[train], config)
        np.testing.assert_allclose(pipeline.named_steps["input_scaler"].mean_, X[train].mean(axis=0))
        expected.append(mean_squared_error(y[valid], pipeline.predict(X[valid])))
    np.testing.assert_allclose(record["fold_mse"], expected, rtol=1e-8)


def test_outer_validation_rows_never_enter_inner_search(monkeypatch):
    X = np.arange(72, dtype=float).reshape(36, 2)
    y = 2 * X[:, 0] + 1
    config = small_config()
    observed = []
    original = pr.search

    def traced_search(X_inner, y_inner, *args, **kwargs):
        observed.append(X_inner[:, 0].copy())
        return original(X_inner, y_inner, *args, **kwargs)

    monkeypatch.setattr(pr, "search", traced_search)
    summary, oof, assignments = pr.nested_evaluate(X, y, config, 1)
    for fold, training_values in enumerate(observed, start=1):
        validation_values = X[assignments == fold, 0]
        assert not np.intersect1d(training_values, validation_values).size
    assert np.isfinite(oof).all()
    assert len(summary["outer_folds"]) == 3


def test_end_to_end_saved_model_submission_and_resume(tmp_path):
    rng = np.random.default_rng(8)
    train = pd.DataFrame(rng.uniform(-1, 1, size=(48, 3)), columns=["x1", "x2", "x3"])
    train["y"] = 1 + 2 * train.x1 + 3 * train.x2 * train.x3
    test = train.drop(columns="y").iloc[:7].copy()
    train.to_csv(tmp_path / "TEST_train_var2.csv", index=False)
    test.to_csv(tmp_path / "TEST_test_var2.csv", index=False)
    config = small_config()
    output = tmp_path / "output"
    pr.train_problem(tmp_path, "TEST", 2, config, output)
    predictions = pd.read_csv(output / "predictions/TEST_pred_var2.csv")
    assert predictions.columns.tolist() == ["y"]
    assert len(predictions) == len(test)
    artifact = joblib.load(output / "var2/model.joblib")
    np.testing.assert_allclose(predictions.y, artifact["pipeline"].predict(test.to_numpy()))
    selection = json.loads((output / "var2/final_selection.json").read_text())
    assert selection["selected"]["degree"] == 2
    pr.train_problem(tmp_path, "TEST", 2, config, output)
    np.testing.assert_allclose(predictions, pd.read_csv(output / "predictions/TEST_pred_var2.csv"))
    config["seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        pr.train_problem(tmp_path, "TEST", 2, config, output)


def test_schema_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x2,x1\n1,2\n")
    with pytest.raises(ValueError, match="expected columns"):
        pr.read_features(path, ["x1", "x2"])
    path.write_text("x1,x2\n1,inf\n")
    with pytest.raises(ValueError, match="finite"):
        pr.read_features(path, ["x1", "x2"])
    with pytest.raises(ValueError, match="finite"):
        pr.write_predictions(tmp_path / "bad_predictions.csv", [np.nan])


def test_degree_ten_count_and_intercept():
    processor = pr.make_preprocessor(10)
    result = processor.fit_transform(np.random.default_rng(1).normal(size=(12, 6)))
    assert result.shape == (12, 8007)


def test_nonconvergent_candidate_is_excluded(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    class Nonconvergent:
        def set_params(self, **kwargs):
            return self

        def fit(self, X, y):
            warnings.warn("Synthetic nonconvergence", ConvergenceWarning)
            return self

    original = pr.make_estimator

    def estimator(candidate, config, warm_start=False):
        if candidate.model == "lasso":
            return Nonconvergent()
        return original(candidate, config, warm_start)

    monkeypatch.setattr(pr, "make_estimator", estimator)
    X = np.random.default_rng(4).normal(size=(30, 2))
    y = X[:, 0] + X[:, 1]
    config = small_config()
    winner, records = pr.search(X, y, config, 1)
    assert winner.model != "lasso"
    assert all(r["mean_mse"] is None and len(r["failures"]) == 2
               for r in records if r["candidate"]["model"] == "lasso")
