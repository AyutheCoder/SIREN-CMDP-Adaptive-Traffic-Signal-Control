"""Dependency-free test runner.

Discovers ``test_*.py`` in this directory and runs every ``test_*`` callable.
Written without pytest so the suite runs in the same bare virtualenv that runs
the experiments.

    python tests/run_tests.py            # everything
    python tests/run_tests.py shield     # only modules matching 'shield'
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in TESTS_DIR.glob("test_*.py") if pattern in p.stem)
    if not files:
        print(f"no test modules matching {pattern!r}")
        return 1

    passed = failed = 0
    failures = []
    t_start = time.time()

    for path in files:
        print(f"\n{path.stem}")
        try:
            module = load_module(path)
        except Exception:
            print(f"  !! import failed")
            traceback.print_exc()
            failed += 1
            failures.append(f"{path.stem} (import)")
            continue

        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
                print(f"  ok    {name}  ({time.time() - t0:.2f}s)")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}")
                traceback.print_exc()
                failed += 1
                failures.append(f"{path.stem}.{name}")

    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed in {time.time() - t_start:.1f}s")
    if failures:
        for f in failures:
            print(f"  FAILED {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
