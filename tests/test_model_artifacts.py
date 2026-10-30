"""Checks that the saved Phase 3 artifacts are internally consistent (targets, split, leakage, baselines, reproducibility).

Run from the project folder:   python tests/test_model_artifacts.py       (or: python -m pytest tests -q)
"""
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("REIA_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
sys.path.insert(0, str(ROOT / "tools"))
import verify_artifacts as V  # noqa: E402


class Skip(Exception):
    pass


def test_saved_artifacts_pass_independent_verification():
    needed = ["models/classifier_pipeline.joblib", "models/regressor_pipeline.joblib", "models/model_metadata.json",
              "phase3_outputs/label_rule.json", "phase3_outputs/split_ids.csv", "outputs/engineered_features.csv"]
    missing = [f for f in needed if not (ROOT / f).exists()]
    if missing:
        raise Skip(f"missing {missing}")
    report = V.verify(ROOT)
    failures = [f"{r['check']}: {r['detail']}" for r in report.failed]
    assert not failures, "verification failures:\n" + "\n".join(failures)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Skip as why:
            print(f"SKIP  {name}: {why}")
        except Exception:
            failed += 1; print(f"FAIL  {name}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
