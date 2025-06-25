"""
Comprehensive tests for mlops.evaluation.evaluation
===================================================

Key points
----------
* Adds repo-root and src/ to ``sys.path`` so both
  ``mlops. …`` **and** ``src.mlops. …`` import styles work.
* Stubs *both* ``load_config`` import paths to avoid the YAML file.
* Monkeys-patches **ModelEvaluator._load_model** once so no test needs a
  real `.pkl` in advance — it just returns a dummy dict.
"""

from __future__ import annotations

# --------------------------------------------------------------------- #
# 0.  Imports & sys.path surgery
# --------------------------------------------------------------------- #
import sys
import matplotlib                 #  <--  add these two lines
matplotlib.use("Agg", force=True)
import json
import pickle
import importlib
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]      # project root
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --------------------------------------------------------------------- #
# 1.  Fake load_config for BOTH possible import paths
# --------------------------------------------------------------------- #
def _fake_load_config(_path):  # always returns an empty config
    return {}

_stub = ModuleType("data_validation_stub")
_stub.load_config = _fake_load_config
sys.modules["mlops.data_validation.data_validation"] = _stub
sys.modules["src.mlops.data_validation.data_validation"] = _stub

# --------------------------------------------------------------------- #
# 2.  Import evaluation module *after* stubbing
# --------------------------------------------------------------------- #
_eval_mod = importlib.import_module("mlops.evaluation.evaluation")
ModelEvaluator  = _eval_mod.ModelEvaluator
generate_report = _eval_mod.generate_report

# --------------------------------------------------------------------- #
# 3.  Monkey-patch ModelEvaluator._load_model so no real file is needed
# --------------------------------------------------------------------- #
setattr(ModelEvaluator, "_load_model", lambda self: {"dummy": "model"})

# --------------------------------------------------------------------- #
# 4.  Fixtures & helpers
# --------------------------------------------------------------------- #
@pytest.fixture
def evaluator(tmp_path: Path) -> ModelEvaluator:
    """A ready-to-use evaluator with a temporary metrics path."""
    return ModelEvaluator(
        model_path=str(tmp_path / "linear.pkl"),          # never opened
        test_data_dir=str(tmp_path),
        config={"artifacts": {"metrics_path": str(tmp_path / "metrics.json")}},
    )


def _write_class_files(tmp: Path, X: pd.DataFrame, y: pd.Series) -> None:
    X.to_csv(tmp / "X_test_class.csv", index=False)
    y.to_csv(tmp / "y_test_class.csv", index=False)


# --------------------------------------------------------------------- #
# 5.  Basic plot tests
# --------------------------------------------------------------------- #
def test_plot_confusion_matrix(evaluator: ModelEvaluator):
    y = pd.Series([0, 1, 0, 1]); preds = np.array([0, 1, 1, 0])
    with mock.patch("mlops.evaluation.evaluation.plt.savefig") as save:
        evaluator.plot_confusion_matrix(y, preds)
        save.assert_called_once()
        assert "confusion_matrix.png" in save.call_args.args[0]


def test_plot_regression_predictions(evaluator: ModelEvaluator):
    df   = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4),
                         "BTCUSDT_price": [100, 101, 102, 103]})
    true = pd.Series([100, 101, 102, 103])
    pred = np.asarray([100.4, 101.2, 102.1, 102.8])
    with mock.patch("mlops.evaluation.evaluation.plt.savefig") as save:
        evaluator.plot_regression_predictions(df, true, pred)
        save.assert_called_once()
        assert "price_prediction_plot.png" in save.call_args.args[0]

# --------------------------------------------------------------------- #
# 6.  save_metrics_report
# --------------------------------------------------------------------- #
def test_save_metrics_report(tmp_path: Path, evaluator: ModelEvaluator):
    evaluator.config["artifacts"]["metrics_path"] = str(tmp_path / "m.json")
    evaluator.save_metrics_report({"rmse": 1.23}, {"accuracy": 0.9})
    data = json.loads((tmp_path / "m.json").read_text())
    assert {"linear_regression", "logistic_regression"} <= data.keys()


def test_save_metrics_report_oserror(tmp_path: Path):
    ev = ModelEvaluator("dummy.pkl", str(tmp_path),
                        {"artifacts": {"metrics_path": str(tmp_path / "m.json")}})
    with mock.patch("builtins.open", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            ev.save_metrics_report({}, {})

# --------------------------------------------------------------------- #
# 7.  generate_report wiring
# --------------------------------------------------------------------- #
def test_generate_report_calls_evaluate_models():
    with mock.patch.object(_eval_mod, "evaluate_models") as evm:
        generate_report({"cfg": 1})
        evm.assert_called_once()

# --------------------------------------------------------------------- #
# 8.  Model loading helpers
# --------------------------------------------------------------------- #
def test_load_model_file_not_found(evaluator: ModelEvaluator):
    with pytest.raises(FileNotFoundError):
        evaluator.load_model("nope.pkl")


def test_load_model_success(tmp_path: Path, evaluator: ModelEvaluator):
    obj = {"hi": "there"}; pkl = tmp_path / "m.pkl"
    pkl.write_bytes(pickle.dumps(obj))
    assert evaluator.load_model(str(pkl)) == obj


def test_load_both_models(tmp_path: Path, evaluator: ModelEvaluator):
    obj = {"ok": True}
    lin = tmp_path / "lin.pkl"; log = tmp_path / "log.pkl"
    for p in (lin, log): p.write_bytes(pickle.dumps(obj))
    evaluator.config["model"] = {
        "linear_regression":   {"save_path": str(lin)},
        "logistic_regression": {"save_path": str(log)},
    }
    m1, m2 = evaluator.load_both_models()
    assert m1 == m2 == obj

# --------------------------------------------------------------------- #
# 9.  evaluate_regression branches
# --------------------------------------------------------------------- #
def test_evaluate_regression_missing_data(tmp_path: Path):
    ev = ModelEvaluator("x.pkl", str(tmp_path), {})
    assert ev.evaluate_regression() == {}


def test_evaluate_regression_predict_error(tmp_path: Path):
    class Bad:                       # raises in predict
        def predict(self, X): raise RuntimeError("boom")
    ev = ModelEvaluator("x.pkl", str(tmp_path), {}); ev.model = Bad()
    assert ev.evaluate_regression() == {}


def test_evaluate_regression_success(monkeypatch, tmp_path: Path):
    class Good:                      # always predicts +1
        def predict(self, X): return np.ones(len(X))
    ev = ModelEvaluator("x.pkl", str(tmp_path), {}); ev.model = Good()
    X = pd.DataFrame({"a": [0, 0, 0]}); y = pd.Series([0, 0, 0])
    monkeypatch.setattr(ev, "_load_test_data", lambda *_: (X, y))
    metrics = ev.evaluate_regression()
    assert pytest.approx(metrics["rmse"]) == 1.0

# --------------------------------------------------------------------- #
# 10.  evaluate_classification branches
# --------------------------------------------------------------------- #
def test_evaluate_classification_no_proba(tmp_path: Path):
    class Plain:
        def predict(self, X): return np.zeros(len(X))

    X = pd.DataFrame({"x": [1, 2]}); y = pd.Series([0, 1])
    _write_class_files(tmp_path, X, y)

    ev = ModelEvaluator("x.pkl", str(tmp_path), {}); ev.model = Plain()
    metrics, *_ = ev.evaluate_classification()
    assert {"accuracy", "f1_score"} <= metrics.keys()


def test_evaluate_classification_proba_error(tmp_path: Path):
    class Bad:
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): raise RuntimeError("fail")

    X = pd.DataFrame({"x": [1, 2]}); y = pd.Series([0, 1])
    _write_class_files(tmp_path, X, y)

    ev = ModelEvaluator("x.pkl", str(tmp_path), {}); ev.model = Bad()
    m, p, s = ev.evaluate_classification()
    assert m == {} and p == {} and s.empty


def _noop(*_args, **_kwargs):         # <-- accepts save_path=…
    """Do nothing; used to silence plotting in tests."""
    return None

# ------------------------------------------------------------------ #
# 10. evaluate_classification branches
# ------------------------------------------------------------------ #
def test_evaluate_classification_with_proba(monkeypatch, tmp_path: Path):
    class Good:
        def predict(self, X): return np.array([0, 1])
        def predict_proba(self, X):
            return np.column_stack([1 - self.predict(X), self.predict(X)])

    # prepare tiny test-set files
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    _write_class_files(tmp_path, X, y)

    ev = ModelEvaluator("x.pkl", str(tmp_path), {})
    ev.model = Good()

    # silence plotting (must accept **kwargs!)
    monkeypatch.setattr(ev, "_plot_confusion_matrix", _noop)
    monkeypatch.setattr(ev, "_plot_roc_curve",        _noop)

    metrics, plots, sample = ev.evaluate_classification()

    assert metrics["accuracy"] == 1.0
    assert "confusion_matrix" in plots
    assert not sample.empty

# --------------------------------------------------------------------- #
# 11.  _load_test_data guard
# --------------------------------------------------------------------- #
def test_load_test_data_missing_files(tmp_path: Path):
    ev = ModelEvaluator("x.pkl", str(tmp_path), {})
    with pytest.raises(FileNotFoundError):
        ev._load_test_data("whatever")

# --------------------------------------------------------------------- #
# 12.  prepare_test_data branches
# --------------------------------------------------------------------- #
class DummyScaler:
    def transform(self, X):  # noqa: N802
        return X.to_numpy(dtype=float)

PIPELINE = {
    "scaler": DummyScaler(),
    "selected_features_reg":   ["f1"],
    "selected_features_class": ["f1"],
    "all_feature_cols":        ["f1"],
}


def _patch_feature_helpers(monkeypatch):
    monkeypatch.setattr(
        "mlops.evaluation.evaluation.define_features_and_label",
        lambda: (["f1"], "BTCUSDT_price"),
    )
    monkeypatch.setattr(
        "mlops.evaluation.evaluation.create_price_direction_label",
        lambda d, _: d.assign(price_direction=[0] * len(d)),
    )
    monkeypatch.setattr(
        "mlops.evaluation.evaluation.prepare_features",
        lambda d, f, l: (d[f], d[l], d["price_direction"]),
    )
    monkeypatch.setattr(
        "mlops.evaluation.evaluation.split_data",
        lambda X, y, **_: (None, X, None, y),
    )


def test_prepare_test_data_no_pipeline(monkeypatch):
    ev = ModelEvaluator("x.pkl", "dir", {}); ev.preprocessing_pipeline = None
    _patch_feature_helpers(monkeypatch)

    df = pd.DataFrame({"timestamp": [1, 2],
                       "f1": [0.1, 0.2],
                       "BTCUSDT_price": [100, 101]})
    Xr, Xc, yr, yc = ev.prepare_test_data(df)
    np.testing.assert_array_equal(Xr, [[0.1], [0.2]])
    assert np.array_equal(Xr, Xc) and list(yr) == [100, 101]


def test_prepare_test_data_with_pipeline(monkeypatch):
    ev = ModelEvaluator("x.pkl", "dir", {}); ev.preprocessing_pipeline = PIPELINE
    _patch_feature_helpers(monkeypatch)

    df = pd.DataFrame({"timestamp": [1, 2, 3, 4],
                       "f1": [0.1, 0.2, 0.3, 0.4],
                       "BTCUSDT_price": [100, 101, 102, 103]})
    Xr, Xc, *_ = ev.prepare_test_data(df)
    assert Xr.shape == (4, 1) and np.array_equal(Xr, Xc)
