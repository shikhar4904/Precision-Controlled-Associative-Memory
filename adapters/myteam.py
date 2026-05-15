"""
Precision-only PCAM agent.

This adapter respects the frozen-model rule: it never modifies PCAMModel,
model_params, R, the gradient, or the integrator. It only returns a positive
64-dimensional precision vector.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.pi_min = float(model_params["pi_min"])
        self.pi_max = float(model_params["pi_max"])
        self.eps = np.finfo(np.float64).eps
        self.precision_log_span = np.log(self.pi_max / self.pi_min)
        self.optim_steps = 2 * self.N
        self.optim_lr = self.precision_log_span / np.sqrt(self.N)
        self.optim_decay = np.exp(np.log(0.5) / max(self.optim_steps, 1))
        self.pattern_norms = np.linalg.norm(self.X, axis=1)
        self.pattern_norms[self.pattern_norms < self.eps] = 1.0
        normalised_x = self.X / self.pattern_norms[:, None]
        pairwise_cos = normalised_x @ normalised_x.T
        np.fill_diagonal(pairwise_cos, -np.inf)
        nearest_neighbour_cos = float(np.max(pairwise_cos))
        self.near_pattern_cos = 0.5 * (1.0 + nearest_neighbour_cos)
        self.query_floor = np.median(np.abs(self.X))
        self.spread_profiles = self._build_spread_profiles()

    def _rank_patterns(self, q: np.ndarray) -> np.ndarray:
        # The corruption process masks coordinates toward zero. Weighting by
        # |q| lets reliable surviving coordinates dominate the first-pass guess.
        reliable = q * np.abs(q)
        return self.X @ reliable

    def _hessian_at(self, pattern: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ pattern)
        z = z - z.max()
        s = np.exp(z)
        s = s / max(s.sum(), self.eps)
        weighted_x = s[:, None] * self.X
        cov = weighted_x.T @ self.X - np.outer(s @ self.X, s @ self.X)
        h = self.R - self.eta * self.beta * cov
        return 0.5 * (h + h.T)

    def _condition_number(self, h: np.ndarray, pi: np.ndarray) -> float:
        pi = np.clip(pi, self.pi_min, self.pi_max)
        pi = pi / max(pi.mean(), self.eps)
        d = np.sqrt(pi)
        s = (d[:, None] * h) * d[None, :]
        eigs = np.linalg.eigvalsh(0.5 * (s + s.T))
        eigs = eigs[eigs > 1e-9]
        if len(eigs) < 2:
            return float("inf")
        return float(eigs[-1] / eigs[0])

    def _improve_spread_profile(self, h: np.ndarray) -> np.ndarray:
        candidates = [np.ones(self.N)]
        diag = np.maximum(np.diag(h), self.eps)
        row = np.maximum(np.sum(np.abs(h), axis=1), self.eps)
        off = np.maximum(row - np.abs(np.diag(h)), self.eps)
        candidates.extend([diag, 1.0 / diag, np.sqrt(diag), 1.0 / np.sqrt(diag)])
        candidates.extend([1.0 / row, 1.0 / off])

        best = min(candidates, key=lambda p: self._condition_number(h, p))
        best_score = self._condition_number(h, best)

        # A short legitimate log-space descent. It is intentionally modest so
        # construction stays fast for larger hidden K values.
        y = np.log(np.clip(best / max(best.mean(), self.eps), self.pi_min, self.pi_max))
        lr = self.optim_lr
        for _ in range(self.optim_steps):
            pi = np.exp(y)
            d = np.sqrt(pi / max(pi.mean(), self.eps))
            s = (d[:, None] * h) * d[None, :]
            vals, vecs = np.linalg.eigh(0.5 * (s + s.T))
            if vals[0] <= 1e-9:
                break
            grad = vecs[:, -1] ** 2 - vecs[:, 0] ** 2
            y -= lr * grad
            y -= y.mean()
            y = np.clip(y, np.log(self.pi_min), np.log(self.pi_max))
            lr *= self.optim_decay
            score = self._condition_number(h, np.exp(y))
            if score < best_score:
                best_score = score
                best = np.exp(y).copy()
        return np.maximum(best, self.eps)

    def _build_spread_profiles(self) -> np.ndarray:
        return np.array([
            self._improve_spread_profile(self._hessian_at(pattern))
            for pattern in self.X
        ])

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        q_norm = np.linalg.norm(q)
        if q_norm > self.eps:
            q = q / q_norm

        scores = self._rank_patterns(q)
        best_idx = int(np.argmax(scores))
        best = self.X[best_idx]

        cosine = float((self.X[best_idx] @ q) / self.pattern_norms[best_idx])
        if cosine >= self.near_pattern_cos:
            return self.spread_profiles[best_idx]

        # Strong selected-pattern coordinates that are missing or small in the
        # query are exactly where precision can help PCAM recover the attractor.
        missing_stroke = np.abs(best) / (np.abs(q) + self.query_floor)
        missing_stroke /= max(missing_stroke.mean(), self.eps)

        # Deliberately allow the harness to do its own clipping and
        # mean-normalisation. Pre-clipping changes the shape and hurts retrieval.
        pi = 1.0 + (missing_stroke - 1.0)
        return np.maximum(pi, self.eps)
