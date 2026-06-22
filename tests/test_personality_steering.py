"""Spec B — toy-decoder steering-hook tests (skip-guarded; torch-only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception:
        print("SKIP test_personality_steering (torch unavailable)")
        return 0
    print("PASS 0 hook tests (toy hook tests added in Task 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
