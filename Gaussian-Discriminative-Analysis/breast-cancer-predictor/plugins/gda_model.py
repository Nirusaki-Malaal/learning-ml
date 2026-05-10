from __future__ import annotations

import numpy as np
import pandas as pd


class GDABreastCancerModel:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        target: pd.Series,
        regularization: float = 1e-3,
    ):
        self.feature_names = list(feature_frame.columns)
        self.regularization = regularization
        self.raw_mean = feature_frame.mean(axis=0).to_numpy(dtype=float)
        self.raw_std = feature_frame.std(axis=0, ddof=0).replace(0, 1).to_numpy(dtype=float)
        self.X = self._standardize(feature_frame.to_numpy(dtype=float))
        self.y = target.to_numpy(dtype=int)
        self.n_features = self.X.shape[1]
        self._fit()

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.raw_mean) / self.raw_std

    def _fit(self) -> None:
        x0 = self.X[self.y == 0]
        x1 = self.X[self.y == 1]
        self.mean0 = x0.mean(axis=0)
        self.mean1 = x1.mean(axis=0)
        self.phi = float(self.y.mean())

        centered = np.empty_like(self.X)
        centered[self.y == 0] = x0 - self.mean0
        centered[self.y == 1] = x1 - self.mean1

        covariance = (centered.T @ centered) / len(self.y)
        covariance += np.eye(self.n_features) * self.regularization
        self.covariance = covariance
        self.covariance_inv = np.linalg.inv(covariance)
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")
        self.covariance_log_det = float(log_det)

    def _feature_array(self, features: dict[str, float]) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        values = np.array([float(features[name]) for name in self.feature_names], dtype=float)
        return self._standardize(values)

    def _log_gaussian(self, x: np.ndarray, mean: np.ndarray) -> float:
        diff = x - mean
        exponent = -0.5 * float(diff @ self.covariance_inv @ diff.T)
        normalizer = -0.5 * (self.n_features * np.log(2 * np.pi) + self.covariance_log_det)
        return float(normalizer + exponent)

    def predict(self, features: dict[str, float]) -> dict:
        x = self._feature_array(features)
        log_benign = np.log(1 - self.phi) + self._log_gaussian(x, self.mean0)
        log_malignant = np.log(self.phi) + self._log_gaussian(x, self.mean1)
        max_log = max(log_benign, log_malignant)
        exp_benign = np.exp(log_benign - max_log)
        exp_malignant = np.exp(log_malignant - max_log)
        total = exp_benign + exp_malignant
        malignant_probability = float(exp_malignant / total)
        benign_probability = float(exp_benign / total)
        prediction = 1 if malignant_probability >= 0.5 else 0
        return {
            "prediction": prediction,
            "label": "Malignant" if prediction == 1 else "Benign",
            "malignant_probability": malignant_probability,
            "benign_probability": benign_probability,
            "scores": {
                "log_benign": float(log_benign),
                "log_malignant": float(log_malignant),
            },
        }

    def accuracy_on_training_data(self) -> float:
        correct = 0
        for row, target in zip(self.X, self.y):
            raw = row * self.raw_std + self.raw_mean
            features = dict(zip(self.feature_names, raw))
            correct += int(self.predict(features)["prediction"] == int(target))
        return correct / len(self.y)

    def class_separation(self) -> list[dict]:
        separation = np.abs(self.mean1 - self.mean0)
        ranked = sorted(
            zip(self.feature_names, separation),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"feature": name, "label": name.replace("_", " ").title(), "score": float(score)}
            for name, score in ranked
        ]
