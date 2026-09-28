"""Warm up Windows Defender/SmartScreen cloud verdicts for all project imports.

The local Application Control policy intermittently blocks first-seen DLLs
until a cloud reputation verdict is cached. Retrying imports a few times
resolves it; this script does that systematically and reports what passes.
"""
from __future__ import annotations

import importlib
import sys
import time

MODULES = [
    "numpy", "pandas", "scipy.stats", "scipy.linalg",
    "sklearn.pipeline", "sklearn.dummy", "sklearn.naive_bayes",
    "sklearn.ensemble", "sklearn.metrics", "sklearn.cluster",
    "sklearn.decomposition", "sklearn.preprocessing",
    "sklearn.model_selection", "statsmodels.api",
    "statsmodels.tsa.holtwinters", "plotly.express", "plotly.graph_objects",
    "matplotlib.pyplot", "seaborn",
]


def try_import(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {mod}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    remaining = list(MODULES)
    for attempt in range(1, 6):
        if not remaining:
            break
        print(f"--- attempt {attempt} ({len(remaining)} module groups left)")
        still = []
        for mod in remaining:
            ok = try_import(mod)
            if not ok:
                still.append(mod)
        remaining = still
        if remaining:
            time.sleep(3)
    if remaining:
        print("STILL BLOCKED:", remaining)
        return 1
    print("All imports OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
