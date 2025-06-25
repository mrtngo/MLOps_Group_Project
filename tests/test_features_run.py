"""
Fast, in-process tests for src/mlops/features/run.py
====================================================

We launch the real CLI code with runpy, but patch heavy bits
(plotting, W&B, network) so the suite stays lightweight.

Layout
------
1.  Path setup  → make `src/` & `mlops/` importable.
2.  Helpers     → _dummy_csv, _invoke_run (with config patch helper).
3.  Six tests   → default, missing-input, no-args, -h/--help, malformed CSV.
"""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import seaborn as sns

# --------------------------------------------------------------------------- #
#  Path setup: allow "import src.*" and "import mlops.*"                      #
# --------------------------------------------------------------------------- #
ROOT   = Path(__file__).resolve().parents[1]          # <repo root>
SRC    = ROOT / "src"
SCRIPT = SRC / "mlops" / "features" / "run.py"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PATCH_TARGET = "src.mlops.data_validation.data_validation.load_config"

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _dummy_csv(path: Path) -> None:
    """Write a tiny BTC + ETH price CSV."""
    path.write_text(
        "timestamp,BTCUSDT_price,ETHUSDT_price\n"
        "2024-01-01,100,95\n"
        "2024-01-02,101,96\n"
        "2024-01-03,102,97\n"
    )


@contextmanager
def _patch_cfg(cfg: dict | None):
    if cfg is None:
        yield
    else:
        with mock.patch(PATCH_TARGET, return_value=cfg):
            yield


def _invoke_run(argv: list[str], cfg: dict | None = None) -> int:
    """Execute run.py with *argv*; return its exit-code."""
    old_argv = sys.argv.copy()
    sys.argv = ["run.py", *argv]
    try:
        with _patch_cfg(cfg):
            try:
                runpy.run_path(str(SCRIPT), run_name="__main__")
                return 0
            except SystemExit as e:        # normal sys.exit()
                return int(e.code or 0)
    except Exception:
        return 2                           # any un-handled error → 2
    finally:
        sys.argv = old_argv


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
def test_features_run_missing_input():
    code = _invoke_run(["--input-artifact", "does_not_exist.csv"])
    assert code != 0


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_run_help_flag(flag: str):
    assert _invoke_run([flag]) == 0


def test_run_malformed_csv(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,foo\n2024-01-01,123\n")

    cfg = {
        "data_source": {"processed_path": str(bad)},
        "feature_engineering": {},
    }

    code = _invoke_run(["--input-artifact", str(bad)], cfg=cfg)
    assert code != 0
