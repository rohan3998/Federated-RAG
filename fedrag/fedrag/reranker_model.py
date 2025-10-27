"""Simple trainable reranker model for FedRAG.

This module implements a tiny linear reranker with SGD using NumPy.
It's designed to be trained federatedly via Flower's Message API.
"""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import numpy as np


def title_similarity(title1: str, title2: str) -> float:
    """Compute fuzzy similarity between two titles.

    Uses a simple token overlap + substring heuristic to avoid heavy deps.
    Returns a float in [0, 1].
    """
    if not title1 or not title2:
        return 0.0
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()
    if not t1 or not t2:
        return 0.0
    if t1 in t2 or t2 in t1:
        return 0.8
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
    return min(1.0, max(0.0, overlap))


class RerankerModel:
    """Tiny linear model: score = w0*x0 + w1*x1 + b.

    Features expected:
      - x0: normalized retrieval score in [0,1] (higher is better)
      - x1: title similarity in [0,1]
    Labels: 0 or 1 (relevant or not).
    """

    def __init__(self, weights: List[float] | np.ndarray | None = None):
        if weights is None:
            # Initialize to prefer retrieval score slightly more than title sim
            # [w0, w1, b]
            self.w = np.array([0.7, 0.3, 0.0], dtype=np.float32)
        else:
            self.w = np.array(weights, dtype=np.float32)
            if self.w.shape != (3,):
                raise ValueError("weights must be length-3 [w0, w1, b]")

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw linear scores."""
        return X @ self.w[:2] + self.w[2]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.predict_scores(X)
        return 1.0 / (1.0 + np.exp(-z))

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Binary cross-entropy loss and gradient w.r.t weights."""
        p = self.predict_proba(X)
        # Clamp to avoid log(0)
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        # Loss
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        # Gradients
        # dL/dz = p - y
        dz = (p - y) / X.shape[0]
        # grads for w0,w1
        grad_w = X.T @ dz
        # grad for bias: sum(dz)
        grad_b = np.sum(dz)
        grad = np.array([grad_w[0], grad_w[1], grad_b], dtype=np.float32)
        return float(loss), grad

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 3) -> None:
        if X.size == 0:
            return
        for _ in range(max(1, int(epochs))):
            _, grad = self.loss_and_grad(X, y)
            self.w -= lr * grad

    def get_weights(self) -> List[float]:
        return [float(v) for v in self.w.tolist()]

    def set_weights(self, weights: List[float]) -> None:
        self.w = np.array(weights, dtype=np.float32)

    @staticmethod
    def save(weights: List[float], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"weights": weights}, f)

    @staticmethod
    def load(filepath: str) -> List[float] | None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            w = data.get("weights")
            if isinstance(w, list) and len(w) == 3:
                return [float(v) for v in w]
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return None


