"""
Overnight training until 7:30 AM. Runs extended physics-informed training in phases.
Saves checkpoints and final model. Designed for maximum ETABS/STAAD/SAP-level accuracy.

Usage:
  cd backend
  python scripts/train_overnight.py [--until 07:30]

Run before bed; will train until target time, then save and exit.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

# Add parent for imports
import sys
_backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend))

# Import train from train_brain (run from backend: python scripts/train_overnight.py)
import importlib.util
_spec = importlib.util.spec_from_file_location("train_brain", _backend / "scripts" / "train_brain.py")
_tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tb)
train = _tb.train
import pandas as pd


def parse_time(s: str) -> tuple[int, int]:
    """Parse HH:MM into (hour, minute)."""
    parts = s.strip().split(":")
    h = int(parts[0]) if len(parts) > 0 else 7
    m = int(parts[1]) if len(parts) > 1 else 30
    return h, m


def should_stop(target_hour: int, target_minute: int) -> bool:
    now = datetime.now()
    if now.hour > target_hour:
        return True
    if now.hour == target_hour and now.minute >= target_minute:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Overnight training until target time")
    parser.add_argument("--until", default="07:30", help="Stop at HH:MM (default 07:30)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    backend = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv) if args.csv else backend / "data" / "etabs_parametric_structural_teacher.csv"
    if not csv_path.exists():
        csv_path = backend.parent / "Previous" / "etabs_parametric_structural_teacher.csv"
    if not csv_path.exists():
        print("ERROR: CSV not found")
        sys.exit(1)

    th, tm = parse_time(args.until)
    n_csv = len(pd.read_csv(csv_path))
    output_path = backend / "models" / "etabs_parametric_structural_brain.pt"

    print("=" * 70)
    print("BALMORES STRUCTURAL — Overnight Physics Training")
    print(f"  Target stop: {th:02d}:{tm:02d}")
    print(f"  CSV: {csv_path} ({n_csv} rows)")
    print("=" * 70)

    phase = 1
    while not should_stop(th, tm):
        print(f"\n>>> Phase {phase} @ {datetime.now().strftime('%H:%M')}")
        train(
            csv_path,
            output_path,
            target_total=n_csv,
            epochs=1200,
            patience=180,
            lr=1e-3,
            width=448,
            head_dim=224,
            use_3res=True,
            use_physics_weights=True,
        )
        phase += 1
        if should_stop(th, tm):
            break
        print("  Waiting 5 min before next phase...")
        time.sleep(300)

    print("\n" + "=" * 70)
    print(f"Training complete @ {datetime.now().strftime('%H:%M')}")
    print(f"Model saved: {output_path}")
    print("=" * 70)
