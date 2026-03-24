"""
10-hour extended physics training: cosine warm-restarts, EMA smoothing, checkpoints.
Run before a long window; saves best + periodic checkpoint to models/checkpoints/.

Usage:
  cd backend
  python scripts/train_10hr.py
  python scripts/train_10hr.py --hours 10 --lr 8e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend))
sys.path.insert(0, str(_backend / "scripts"))

import pandas as pd

import train_brain as tb


def main() -> None:
    parser = argparse.ArgumentParser(description="10h extended brain training")
    parser.add_argument("--hours", type=float, default=10.0, help="Wall-clock training duration")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--checkpoint-mins", type=float, default=30.0)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _backend / "data" / "etabs_parametric_structural_teacher.csv"
        if not csv_path.exists():
            csv_path = _backend.parent / "Previous" / "etabs_parametric_structural_teacher.csv"
    if not csv_path.exists():
        print("ERROR: CSV not found")
        sys.exit(1)

    n_csv = len(pd.read_csv(csv_path))
    out = _backend / "models" / "etabs_parametric_structural_brain.pt"
    ckpt_dir = _backend / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "brain_10hr_latest.pt"

    wall = max(60.0, args.hours * 3600.0)
    print("=" * 70)
    print("BALMORES STRUCTURAL — 10h extended training")
    print(f"  Duration: {wall/3600:.2f} h | CSV rows: {n_csv} | cosine warm-restarts + EMA")
    print("=" * 70)

    tb.train(
        csv_path,
        out,
        target_total=n_csv,
        epochs=9_999_999,
        patience=9_999_999,
        lr=args.lr,
        batch_size=args.batch,
        weight_decay=1.2e-4,
        use_3res=True,
        use_physics_weights=True,
        use_huber=True,
        max_wall_seconds=wall,
        checkpoint_path=ckpt,
        checkpoint_interval_sec=max(300.0, args.checkpoint_mins * 60.0),
        scheduler_mode="cosine_warm",
        ema_decay=0.999,
    )
    print("\nDone. Main model:", out)
    print("Latest checkpoint:", ckpt)
    print("Push: git add backend/models/*.pt && git commit -m \"10h brain\" && git push")


if __name__ == "__main__":
    main()
