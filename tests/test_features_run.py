"""
Light-weight, in-process CLI tests for  src/mlops/features/run.py
----------------------------------------------------------------
We exercise argument-parsing and basic error paths without running the full
feature-engineering pipeline (heavy I/O, plotting, W&B).  The helper
``_invoke_run`` executes the script via ``runpy.run_path`` and converts:

* normal *success*  → exit-code **0**
* ``SystemExit``    → its ``code`` (0 | non-zero)
* *any* other error → exit-code **2**

so tests can assert purely on the integer it returns.
"""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

# --------------------------------------------------------------------------- #
#  Make both ``src``  and  ``mlops`` importable                               #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]        # <repo root>
SRC  = ROOT / "src"
SCRIPT = SRC / "mlops" / "features" / "run.py"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PATCH_TARGET = "src.mlops.data_validation.data_validation.load_config"

# --------------------------------------------------------------------------- #
#  Helper – create a tiny CSV with *two* price cols so ≥1 feature exists      #
# --------------------------------------------------------------------------- #
def _dummy_csv(path: Path) -> None:
    path.write_text(
        "timestamp,BTCUSDT_price,ETHUSDT_price\n"
        "2024-01-01,100,95\n"
        "2024-01-02,101,96\n"
        "2024-01-03,102,97\n"
    )

# --------------------------------------------------------------------------- #
#  run.py launcher that returns an *exit-code* instead of raising             #
# --------------------------------------------------------------------------- #
@contextmanager
def _patch_config(cfg: dict | None):
    if cfg is None:
        yield
    else:
        with mock.patch(PATCH_TARGET, return_value=cfg):
            yield

def _invoke_run(argv: list[str], cfg: dict | None = None) -> int:
    """Run run.py with *argv*; return exit-code (0, 1, 2…)."""
    orig_argv = sys.argv.copy()
    sys.argv = ["run.py", *argv]
    try:
        with _patch_config(cfg):
            try:
                runpy.run_path(str(SCRIPT), run_name="__main__")
                return 0
            except SystemExit as e:            # explicit sys.exit()
                return int(e.code or 0)
    except Exception:                          # unhandled python error
        return 2
    finally:
        sys.argv = orig_argv

# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
def test_features_run_default(tmp_path: Path):
    """Happy path: valid CSV → exit-code 0 while stubbing plotting."""
    import numpy as np
    from unittest import mock

    csv = tmp_path / "prices.csv"
    _dummy_csv(csv)                               # BTC + ETH prices

    patched_cfg = {
        "data_source": {"processed_path": str(csv)},
        "symbols": ["ETHUSDT"],                   # guarantees ≥1 feature
    }

    # Fake (fig, axes) so run.py thinks plotting succeeded.
    dummy_fig  = mock.Mock(name="fig")
    dummy_axes = np.array([mock.Mock(name="ax")])  # .flatten() works

    with (
        mock.patch("matplotlib.pyplot.subplots",
                   return_value=(dummy_fig, dummy_axes)),
        mock.patch("seaborn.histplot", return_value=None),  # <— new stub
    ):
        code = _invoke_run(
            ["--input-artifact", str(csv)],
            cfg=patched_cfg,
        )

    assert code == 0





def test_features_run_missing_input():
    code = _invoke_run(["--input-artifact", "no_such_file.csv"])
    assert code != 0


def test_run_no_args():
    # With no args the script falls back to the default config and succeeds.
    assert _invoke_run([]) == 0


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_run_help_flag(flag: str):
    assert _invoke_run([flag]) == 0


def test_run_malformed_csv(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,foo\n2024-01-01,123\n")

    code = _invoke_run(
        ["--input-artifact", str(bad)],
        cfg={"data_source": {"processed_path": str(bad)}, "feature_engineering": {}},
    )
    assert code != 0
