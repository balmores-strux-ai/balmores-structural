from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(x + residual)


class StructuralBrainModel(nn.Module):
    def __init__(self, in_dim: int, width: int, out_dim: int, head_dim: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.res1 = ResidualBlock(width)
        self.res2 = ResidualBlock(width)
        self.head = nn.Sequential(
            nn.Linear(width, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


class SimpleScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_state(cls, state: Dict[str, List[float]]) -> "SimpleScaler":
        mean = np.array(state["mean"], dtype=np.float64)
        std = np.array(state["std"], dtype=np.float64)
        std[std < 1e-12] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class BrainBundle:
    def __init__(self, model_path: Path) -> None:
        bundle = torch.load(model_path, map_location="cpu")
        self.feature_columns: List[str] = bundle["feature_columns"]
        self.target_columns: List[str] = bundle["target_columns"]
        self.training_ranges: Dict[str, List[float]] = bundle.get("training_ranges", {})
        self.metrics: Dict = bundle.get("metrics", {})
        self.dataset_rows: int = int(bundle.get("dataset_rows", 0))
        self.config: Dict = bundle.get("config", {})

        state = bundle["model_state_dict"]
        in_dim = state["embed.0.weight"].shape[1]
        width = state["embed.0.weight"].shape[0]
        head_dim = state["head.0.weight"].shape[0]
        out_dim = state["head.2.weight"].shape[0]

        self.model = StructuralBrainModel(in_dim, width, out_dim, head_dim)
        self.model.load_state_dict(state)
        self.model.eval()

        self.x_scaler = SimpleScaler.from_state(bundle["x_scaler"])
        self.y_scaler = SimpleScaler.from_state(bundle["y_scaler"])

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        raw = np.array([float(feature_dict.get(c, 0.0)) for c in self.feature_columns], dtype=np.float64).reshape(1, -1)
        raw = np.maximum(raw, 1e-12)
        x_log = np.log(raw)
        x_scaled = self.x_scaler.transform(x_log).astype(np.float32)
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
        pred_log = self.y_scaler.inverse_transform(pred_scaled)
        pred = np.exp(pred_log).reshape(-1)
        out = {self.target_columns[i]: float(pred[i]) for i in range(len(self.target_columns))}
        return out


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "etabs_parametric_structural_brain.pt"
BRAIN = BrainBundle(MODEL_PATH)
