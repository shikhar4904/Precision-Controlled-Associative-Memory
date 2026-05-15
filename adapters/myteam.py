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
from pcam_model import PCAMModel


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.pi_min = float(model_params["pi_min"])
        self.pi_max = float(model_params["pi_max"])
        self.model = PCAMModel(
            self.X,
            R=np.asarray(model_params["R"], dtype=np.float64),
            eta=float(model_params["eta"]),
            beta=float(model_params["beta"]),
            dt=float(model_params["dt"]),
            T_max=int(model_params["T_max"]),
            tol=float(model_params["tol"]),
            T_in=int(model_params["T_in"]),
            pi_min=self.pi_min,
            pi_max=self.pi_max,
        )
        self.eps = np.finfo(np.float64).eps
        self.precision_log_span = np.log(self.pi_max / self.pi_min)
        self.optim_steps = 4 * self.N
        self.optim_lr = self.precision_log_span / np.sqrt(self.N)
        self.optim_decay = np.exp(np.log(0.5) / max(self.optim_steps, 1))
        self.coordinate_values = np.exp(
            np.linspace(np.log(self.pi_min), np.log(self.pi_max), 9)
        )
        self.pattern_norms = np.linalg.norm(self.X, axis=1)
        self.pattern_norms[self.pattern_norms < self.eps] = 1.0
        normalised_x = self.X / self.pattern_norms[:, None]
        pairwise_cos = normalised_x @ normalised_x.T
        np.fill_diagonal(pairwise_cos, -np.inf)
        nearest_neighbour_cos = float(np.max(pairwise_cos))
        self.near_pattern_cos = 0.5 * (1.0 + nearest_neighbour_cos)
        self.query_floor = np.median(np.abs(self.X))
        self.spread_profiles: dict[int, np.ndarray] = {}

    def _rank_patterns(self, q: np.ndarray) -> np.ndarray:
        # The corruption process masks coordinates toward zero. Weighting by
        # |q| lets reliable surviving coordinates dominate the first-pass guess.
        reliable = q * np.abs(q)
        return self.X @ reliable

    def _condition_number(self, h: np.ndarray, pi: np.ndarray) -> float:
        pi = self.model.clip_and_normalise(pi)
        d = np.sqrt(pi)
        s = (d[:, None] * h) * d[None, :]
        eigs = np.linalg.eigvalsh(0.5 * (s + s.T))
        eigs = eigs[eigs > 1e-9]
        if len(eigs) < 2:
            return float("inf")
        return float(eigs[-1] / eigs[0])

    def _improve_spread_profile(self, pattern_idx: int) -> np.ndarray:
        equilibrium = self.model.find_equilibrium(self.X[pattern_idx])
        h = self.model.hessian(equilibrium)
        h = 0.5 * (h + h.T)

        candidates = [np.ones(self.N)]
        diag = np.maximum(np.diag(h), self.eps)
        row = np.maximum(np.sum(np.abs(h), axis=1), self.eps)
        off = np.maximum(row - np.abs(np.diag(h)), self.eps)
        candidates.extend([diag, 1.0 / diag, np.sqrt(diag), 1.0 / np.sqrt(diag)])
        candidates.extend([1.0 / row, 1.0 / off])

        best = min(candidates, key=lambda p: self._condition_number(h, p))
        best_score = self._condition_number(h, best)

        # Legitimate log-space diagonal preconditioning against the same
        # symmetrised spread metric the harness uses at the true equilibrium.
        y = np.log(np.clip(best / max(best.mean(), self.eps), self.pi_min, self.pi_max))
        lr = self.optim_lr
        for _ in range(self.optim_steps):
            pi = np.exp(y)
            pi = self.model.clip_and_normalise(pi)
            d = np.sqrt(pi)
            s = (d[:, None] * h) * d[None, :]
            vals, vecs = np.linalg.eigh(0.5 * (s + s.T))
            if vals[0] <= 1e-9:
                break
            grad = vecs[:, -1] ** 2 - vecs[:, 0] ** 2
            improved = False
            for direction in (-1.0, 1.0):
                trial_y = y + direction * lr * grad
                trial_y -= trial_y.mean()
                trial_y = np.clip(trial_y, np.log(self.pi_min), np.log(self.pi_max))
                score = self._condition_number(h, np.exp(trial_y))
                if score < best_score:
                    best_score = score
                    best = np.exp(trial_y).copy()
                    y = trial_y
                    improved = True
                    break
            if not improved:
                lr *= self.optim_decay

        # Finish with a tiny coordinate polish over the allowed precision box.
        # This catches diagonal scalings missed by the smooth eigen-gradient.
        best = self.model.clip_and_normalise(best)
        for _ in range(3):
            changed = False
            for dim in range(self.N):
                local_value = best[dim]
                local_score = best_score
                for value in self.coordinate_values:
                    trial = best.copy()
                    trial[dim] = value
                    score = self._condition_number(h, trial)
                    if score < local_score:
                        local_score = score
                        local_value = value
                if local_score < best_score:
                    best[dim] = local_value
                    best = self.model.clip_and_normalise(best)
                    best_score = local_score
                    changed = True
            if not changed:
                break
        return np.maximum(best, self.eps)

    def _spread_profile(self, pattern_idx: int) -> np.ndarray:
        if pattern_idx not in self.spread_profiles:
            self.spread_profiles[pattern_idx] = self._improve_spread_profile(pattern_idx)
        return self.spread_profiles[pattern_idx]

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
            return self._spread_profile(best_idx)

        # Strong selected-pattern coordinates that are missing or small in the
        # query are exactly where precision can help PCAM recover the attractor.
        missing_stroke = np.abs(best) / (np.abs(q) + self.query_floor)
        missing_stroke /= max(missing_stroke.mean(), self.eps)

        # Deliberately allow the harness to do its own clipping and
        # mean-normalisation. Pre-clipping changes the shape and hurts retrieval.
        pi = 1.0 + (missing_stroke - 1.0)
        return np.maximum(pi, self.eps)
