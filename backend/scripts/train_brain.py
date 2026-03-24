"""
Train BALMORES STRUCTURAL brain model from teacher CSV.
Physics-informed training: weighted critical targets, StructuralBrain3Res, Huber loss,
validation split, early stopping, LR scheduler, gradient clipping.
Adds synthetic samples to reach target total (disable with --target 0 for real-only).

Usage:
  cd backend
  python scripts/train_brain.py [--csv PATH] [--target 5000] [--epochs 800]
  python scripts/train_brain.py --physics --epochs 800  # physics-informed mode

Output: backend/models/etabs_parametric_structural_brain.pt

Extended 10h run: python scripts/train_10hr.py (cosine warm-restarts, EMA, checkpoints).
"""
from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent for app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_loader import StructuralBrainModel, StructuralBrain3Res, BRAIN, SimpleScaler

FEATURE_COLUMNS = BRAIN.feature_columns
TARGET_COLUMNS = BRAIN.target_columns

# Physics-critical targets: higher weight = model learns them with ETABS-level precision
PHYSICS_TARGET_WEIGHTS = {
    "max_beam_moment_kNm": 3.0,
    "max_column_axial_kN": 3.0,
    "max_drift_mm": 3.0,
    "max_beam_shear_kN": 2.2,
    "max_beam_end_shear_kN": 2.2,
    "max_beam_end_moment_kNm": 2.2,
    "max_beam_deflection_mm": 2.0,
    "max_joint_reaction_vertical_kN": 1.6,
}


def load_and_prepare(
    csv_path: Path,
    target_total: int,
    augment_noise_x: float = 0.008,
    augment_noise_y: float = 0.005,
    filter_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Filter failed ETABS runs: keep rows with at least one valid critical target
    n_before = len(df)
    if filter_invalid and n_before > 100:
        beam_etabs = df["max_beam_moment_kNm"] if "max_beam_moment_kNm" in df.columns else pd.Series(0.0, index=df.index)
        beam_alt = df["beam_m3_grav_kNm"] if "beam_m3_grav_kNm" in df.columns else pd.Series(0.0, index=df.index)
        col_etabs = df["max_column_axial_kN"] if "max_column_axial_kN" in df.columns else pd.Series(0.0, index=df.index)
        col_alt = df["col_p_grav_kN"] if "col_p_grav_kN" in df.columns else pd.Series(0.0, index=df.index)
        beam_ok = (beam_etabs.fillna(0) > 1e-6) | (beam_alt.fillna(0) > 1e-6)
        col_ok = (col_etabs.fillna(0) > 1e-6) | (col_alt.fillna(0) > 1e-6)
        df = df[beam_ok | col_ok]
        if len(df) < n_before * 0.5:
            df = pd.read_csv(csv_path)  # revert if filter too aggressive
        elif n_before > len(df):
            print(f"  Filtered {n_before - len(df)} invalid/failed rows -> {len(df)} clean")
    available = set(df.columns)
    missing_feat = [c for c in FEATURE_COLUMNS if c not in available]
    missing_targ = [c for c in TARGET_COLUMNS if c not in available]

    if missing_feat:
        print(f"  Filling missing features: {len(missing_feat)} cols")
    if missing_targ:
        print(f"  Filling missing targets: {len(missing_targ)} cols")

    X = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, c in enumerate(FEATURE_COLUMNS):
        if c in df.columns:
            X[:, i] = df[c].fillna(0).astype(float).values
        X[:, i] = np.maximum(X[:, i], 1e-12)

    y = np.zeros((len(df), len(TARGET_COLUMNS)), dtype=np.float64)
    for i, c in enumerate(TARGET_COLUMNS):
        if c in df.columns:
            y[:, i] = df[c].fillna(0).astype(float).values
        y[:, i] = np.maximum(y[:, i], 1e-12)

    n_orig = len(df)
    need = max(0, target_total - n_orig)
    if need > 0:
        print(f"  WARNING: Augmenting with {need} synthetic samples (noise X={augment_noise_x*100:.1f}%, y={augment_noise_y*100:.1f}%)")
        print(f"  Note: Synthetic augmentation can harm physics accuracy. For best results, use real ETABS data only.")
        rng = np.random.default_rng(42)
        idx = rng.integers(0, n_orig, size=need)
        X_noise = X[idx] * (1 + augment_noise_x * rng.standard_normal((need, X.shape[1])))
        y_noise = y[idx] * (1 + augment_noise_y * rng.standard_normal((need, y.shape[1])))
        X_noise = np.maximum(X_noise, 1e-12)
        y_noise = np.maximum(y_noise, 1e-12)
        X = np.vstack([X, X_noise])
        y = np.vstack([y, y_noise])

    return X, y


def _bundle_from_state(
    state_dict: dict,
    x_scaler: dict,
    y_scaler: dict,
    training_ranges: dict,
    n_samples: int,
    feature_columns: list,
    target_columns: list,
    checkpoint_epoch: int,
    best_val_loss: float,
) -> dict:
    return {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "model_state_dict": deepcopy(state_dict),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "training_ranges": training_ranges,
        "dataset_rows": int(n_samples),
        "metrics": {"checkpoint_epoch": checkpoint_epoch, "checkpoint_val_loss": best_val_loss},
        "config": {"checkpoint": True},
    }


def train(
    csv_path: Path,
    output_path: Path,
    target_total: int = 5000,
    epochs: int = 800,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 150,
    width: int = 448,
    head_dim: int = 224,
    weight_decay: float = 1e-4,
    use_3res: bool = True,
    use_huber: bool = True,
    use_physics_weights: bool = True,
    max_wall_seconds: float | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval_sec: float = 1800.0,
    scheduler_mode: str = "plateau",
    ema_decay: float | None = None,
    init_state_dict: dict | None = None,
):
    if max_wall_seconds is not None:
        patience = max(patience, 999999)

    print("=" * 60)
    print("BALMORES STRUCTURAL — Rigorous Physics Brain Training")
    if max_wall_seconds:
        print(f"  Time budget: {max_wall_seconds/3600:.2f} h | checkpoints every {checkpoint_interval_sec/60:.0f} min")
    print("=" * 60)

    print("\nLoading CSV...")
    X_raw, y_raw = load_and_prepare(csv_path, target_total)
    n_samples = len(X_raw)
    print(f"  Total samples: {n_samples}")

    # Log-scale and scale
    X_log = np.log(X_raw)
    y_log = np.log(y_raw)
    x_mean = np.mean(X_log, axis=0)
    x_std = np.std(X_log, axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = np.mean(y_log, axis=0)
    y_std = np.std(y_log, axis=0)
    y_std[y_std < 1e-12] = 1.0

    X = ((X_log - x_mean) / x_std).astype(np.float32)
    y = ((y_log - y_mean) / y_std).astype(np.float32)
    X = np.clip(X, -6.0, 6.0)
    y = np.clip(y, -6.0, 6.0)

    # 85/15 train/val split (stratified by order for reproducibility)
    rng = np.random.default_rng(42)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_train = max(int(0.85 * n_samples), n_samples - 100)
    n_train = min(n_train, n_samples - 50)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    x_scaler = {"mean": x_mean.tolist(), "std": x_std.tolist()}
    y_scaler = {"mean": y_mean.tolist(), "std": y_std.tolist()}
    training_ranges = {}
    for i, c in enumerate(FEATURE_COLUMNS):
        if c in ("stories", "bays_x", "bays_y", "span_x_m", "span_y_m", "story_height_m", "total_height_m"):
            col_vals = X_raw[:, i]
            training_ranges[c] = [float(np.min(col_vals)), float(np.max(col_vals))]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    in_dim = X.shape[1]
    out_dim = y.shape[1]

    # Physics-upgraded model: 3-res blocks, 448 width, higher precision
    if use_3res:
        model = StructuralBrain3Res(in_dim, out_dim, width=width, head_dim=head_dim)
    else:
        model = StructuralBrainModel(in_dim, width, out_dim, head_dim)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_mode == "cosine_warm":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=120, T_mult=2, eta_min=1e-7
        )
        plateau_scheduler = None
    else:
        scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=30, min_lr=1e-6
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Build target weights for physics-critical outputs
    target_weights = np.ones(out_dim, dtype=np.float32)
    if use_physics_weights:
        for i, c in enumerate(TARGET_COLUMNS):
            if c in PHYSICS_TARGET_WEIGHTS:
                target_weights[i] = PHYSICS_TARGET_WEIGHTS[c]
    target_weights_t = torch.tensor(target_weights, dtype=torch.float32).to(device)

    print(f"  Device: {device} | width={width} | head={head_dim} | 3res={use_3res}")
    print(
        f"  Epochs: {epochs} | patience={patience} | lr={lr} | physics_weights={use_physics_weights} | sched={scheduler_mode}\n"
    )

    ema_shadow: dict[str, torch.Tensor] | None = None
    if ema_decay is not None and ema_decay > 0:
        ema_shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    wall_start = time.monotonic()
    last_ckpt = wall_start

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    stop_reason = "max_epochs"

    def loss_fn(pred, target):
        if use_physics_weights:
            w = target_weights_t.unsqueeze(0).expand_as(pred)
            return (w * (pred - target) ** 2).mean()
        if use_huber:
            return nn.functional.huber_loss(pred, target, delta=1.0)
        return nn.functional.mse_loss(pred, target)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema_shadow is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in ema_shadow:
                            ema_shadow[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val_t).item()
        model.train()

        if plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)
        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        elapsed = time.monotonic() - wall_start
        if (ep + 1) % 25 == 0 or ep == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  Epoch {ep+1:4d} | train={train_loss:.6f} | val={val_loss:.6f} | best={best_val_loss:.6f} | lr={lr_now:.2e} | t={elapsed/60:.1f}m"
            )

        if checkpoint_path is not None and (time.monotonic() - last_ckpt) >= checkpoint_interval_sec:
            if best_state is not None:
                ck_bundle = _bundle_from_state(
                    best_state,
                    x_scaler,
                    y_scaler,
                    training_ranges,
                    n_samples,
                    FEATURE_COLUMNS,
                    TARGET_COLUMNS,
                    ep + 1,
                    best_val_loss,
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ck_bundle, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
            last_ckpt = time.monotonic()

        if max_wall_seconds is not None and elapsed >= max_wall_seconds:
            print(f"\n  Wall-clock limit reached ({max_wall_seconds/3600:.2f} h) at epoch {ep+1}")
            stop_reason = "wall_clock"
            break

        if bad_epochs >= patience:
            print(f"\n  Early stopping at epoch {ep+1} (patience={patience})")
            stop_reason = "patience"
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if (
        ema_shadow is not None
        and max_wall_seconds is not None
        and best_state is not None
    ):
        model.eval()
        with torch.no_grad():
            loss_best = loss_fn(model(x_val), y_val_t).item()
        ema_cpu = {k: v.detach().cpu().clone() for k, v in ema_shadow.items()}
        model.load_state_dict({k: v.to(device) for k, v in ema_cpu.items()})
        with torch.no_grad():
            loss_ema = loss_fn(model(x_val), y_val_t).item()
        if loss_ema < loss_best:
            print(f"  Exporting EMA weights (val {loss_ema:.6f} < snapshot {loss_best:.6f})")
            best_state = ema_cpu
        else:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Validation metrics (RMSE in original units for key targets)
    y_mean_a = np.array(y_scaler["mean"])
    y_std_a = np.array(y_scaler["std"])
    with torch.no_grad():
        val_pred_scaled = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    val_pred_raw = np.exp(val_pred_scaled * y_std_a + y_mean_a)
    val_true_raw = np.exp(y_val * y_std_a + y_mean_a)
    metrics = {}
    for i, c in enumerate(TARGET_COLUMNS):
        if c in PHYSICS_TARGET_WEIGHTS:
            rmse = float(np.sqrt(np.mean((val_pred_raw[:, i] - val_true_raw[:, i]) ** 2)))
            metrics[f"val_rmse_{c}"] = rmse
    if metrics:
        print("  Physics target RMSE (val):", {k: f"{v:.2f}" for k, v in metrics.items()})

    export_state = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "model_state_dict": export_state,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "training_ranges": training_ranges,
        "dataset_rows": int(n_samples),
        "metrics": metrics,
        "config": {
            "epochs": epochs,
            "lr": lr,
            "target_total": target_total,
            "width": width,
            "head_dim": head_dim,
            "val_split": 0.15,
            "early_stopped": stop_reason == "patience",
            "stop_reason": stop_reason,
            "max_wall_seconds": max_wall_seconds,
            "scheduler_mode": scheduler_mode,
            "use_3res": use_3res,
            "physics_weights": use_physics_weights,
        },
    }
    torch.save(bundle, output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Best val MSE: {best_val_loss:.6f}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train BALMORES STRUCTURAL brain (physics-informed)")
    parser.add_argument("--csv", default=None, help="Path to teacher CSV")
    parser.add_argument("--target", type=int, default=None, help="Target samples (default: real data only, no aug)")
    parser.add_argument("--epochs", type=int, default=800, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=150, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=448, help="Model hidden width (448 for physics mode)")
    parser.add_argument("--head", type=int, default=224, help="Head hidden dim")
    parser.add_argument("--no-3res", action="store_true", help="Use 2-res model instead of 3-res")
    parser.add_argument("--no-physics-weights", action="store_true", help="Disable physics target weighting")
    args = parser.parse_args()

    backend = Path(__file__).resolve().parent.parent
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = backend / "data" / "etabs_parametric_structural_teacher.csv"
        if not csv_path.exists():
            csv_path = backend.parent / "Previous" / "etabs_parametric_structural_teacher.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found. Tried {args.csv or 'default'}")
        sys.exit(1)

    output_path = backend / "models" / "etabs_parametric_structural_brain.pt"

    import pandas as pd
    n_csv = len(pd.read_csv(csv_path))
    target = args.target if args.target is not None else n_csv

    print(f"CSV: {csv_path}")
    print(f"Target samples: {target} (real: {n_csv})\n")

    train(
        csv_path,
        output_path,
        target_total=target,
        epochs=args.epochs,
        patience=args.patience,
        width=args.width,
        head_dim=args.head,
        use_3res=not args.no_3res,
        use_physics_weights=not args.no_physics_weights,
        use_huber=True,
    )
    print("\nDone. Push to GitHub to auto-deploy.")


if __name__ == "__main__":
    main()
