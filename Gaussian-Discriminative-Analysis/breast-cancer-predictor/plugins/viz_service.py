from __future__ import annotations

import numpy as np

from plugins.data_service import BreastCancerData
from plugins.gda_model import GDABreastCancerModel


class VisualizationService:
    def __init__(self, data: BreastCancerData, model: GDABreastCancerModel):
        self.data = data
        self.model = model

    def build(self, features: dict[str, float], x_feature: str, y_feature: str, grid_size: int) -> dict:
        if x_feature not in self.data.feature_names:
            x_feature = "radius_mean"
        if y_feature not in self.data.feature_names:
            y_feature = "texture_mean"
        if x_feature == y_feature:
            y_feature = "texture_mean" if x_feature != "texture_mean" else "radius_mean"

        grid_size = max(12, min(int(grid_size), 45))
        base_features = self.data.median_features()
        base_features.update({k: float(v) for k, v in features.items() if k in base_features})

        df = self.data.df
        x_min, x_max = float(df[x_feature].min()), float(df[x_feature].max())
        y_min, y_max = float(df[y_feature].min()), float(df[y_feature].max())
        x_values = np.linspace(x_min, x_max, grid_size)
        y_values = np.linspace(y_min, y_max, grid_size)

        surface = []
        boundary_points = []
        for y in y_values:
            row = []
            for x in x_values:
                point = dict(base_features)
                point[x_feature] = float(x)
                point[y_feature] = float(y)
                probability = self.model.predict(point)["malignant_probability"]
                row.append(probability)
                if 0.47 <= probability <= 0.53:
                    boundary_points.append({"x": float(x), "y": float(y)})
            surface.append(row)

        scatter = [
            {
                "x": float(row[x_feature]),
                "y": float(row[y_feature]),
                "diagnosis": row["diagnosis"],
                "label": "Malignant" if row["diagnosis"] == "M" else "Benign",
            }
            for _, row in df.iterrows()
        ]

        return {
            "x_feature": x_feature,
            "y_feature": y_feature,
            "x_values": x_values.tolist(),
            "y_values": y_values.tolist(),
            "surface": surface,
            "boundary": boundary_points,
            "scatter": scatter,
            "selected_point": {
                "x": float(base_features[x_feature]),
                "y": float(base_features[y_feature]),
            },
            "ellipses": self._ellipses(x_feature, y_feature),
            "distribution": self._distribution(x_feature),
            "separation": self.model.class_separation()[:10],
        }

    def _ellipses(self, x_feature: str, y_feature: str) -> dict:
        """Compute elliptical contour parameters for both classes in the 2D subspace."""
        df = self.data.df
        xi = self.data.feature_names.index(x_feature)
        yi = self.data.feature_names.index(y_feature)

        result = {}
        for label, diag_code in [("benign", "B"), ("malignant", "M")]:
            subset = df[df["diagnosis"] == diag_code]
            x_vals = subset[x_feature].to_numpy(dtype=float)
            y_vals = subset[y_feature].to_numpy(dtype=float)

            mean_x = float(np.mean(x_vals))
            mean_y = float(np.mean(y_vals))

            cov = np.cov(x_vals, y_vals)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

            contours = []
            for n_std in [1.0, 2.0]:
                width = 2.0 * n_std * float(np.sqrt(eigenvalues[0]))
                height = 2.0 * n_std * float(np.sqrt(eigenvalues[1]))
                points = []
                for t in np.linspace(0, 2 * np.pi, 60):
                    px = (width / 2) * np.cos(t)
                    py = (height / 2) * np.sin(t)
                    cos_a = np.cos(np.radians(angle))
                    sin_a = np.sin(np.radians(angle))
                    rx = mean_x + px * cos_a - py * sin_a
                    ry = mean_y + px * sin_a + py * cos_a
                    points.append({"x": float(rx), "y": float(ry)})
                contours.append({"std": n_std, "points": points})

            result[label] = {
                "mean": {"x": mean_x, "y": mean_y},
                "angle": angle,
                "contours": contours,
            }

        return result

    def _distribution(self, feature: str) -> dict:
        df = self.data.df
        values = df[feature]
        bins = np.linspace(float(values.min()), float(values.max()), 18)
        benign_counts, edges = np.histogram(df[df["diagnosis"] == "B"][feature], bins=bins)
        malignant_counts, _ = np.histogram(df[df["diagnosis"] == "M"][feature], bins=bins)
        labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
        return {
            "feature": feature,
            "labels": labels,
            "benign": benign_counts.astype(int).tolist(),
            "malignant": malignant_counts.astype(int).tolist(),
        }

    def confusion_matrix(self, features_override: dict[str, float] | None = None) -> dict:
        """Compute confusion matrix on training data."""
        df = self.data.df
        tp = tn = fp = fn = 0
        for _, row in df.iterrows():
            features = {name: float(row[name]) for name in self.data.feature_names}
            pred = self.model.predict(features)["prediction"]
            actual = 1 if row["diagnosis"] == "M" else 0
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            elif pred == 1 and actual == 0:
                fp += 1
            else:
                fn += 1
        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round((tp + tn) / total, 4) if total else 0,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total": total,
        }

    def correlation_matrix(self, top_n: int = 8) -> dict:
        """Return correlation matrix for the top N most separating features."""
        top_features = [item["feature"] for item in self.model.class_separation()[:top_n]]
        df = self.data.df[top_features]
        corr = df.corr()
        labels = [name.replace("_", " ").title() for name in top_features]
        matrix = corr.values.tolist()
        return {"labels": labels, "features": top_features, "matrix": matrix}

    def class_statistics(self) -> dict:
        """Per-class summary statistics for all features."""
        df = self.data.df
        benign = df[df["diagnosis"] == "B"]
        malignant = df[df["diagnosis"] == "M"]
        stats = []
        for name in self.data.feature_names:
            stats.append({
                "feature": name,
                "label": name.replace("_", " ").title(),
                "benign_mean": round(float(benign[name].mean()), 4),
                "benign_std": round(float(benign[name].std()), 4),
                "malignant_mean": round(float(malignant[name].mean()), 4),
                "malignant_std": round(float(malignant[name].std()), 4),
                "diff_pct": round(
                    abs(float(malignant[name].mean()) - float(benign[name].mean()))
                    / max(float(benign[name].mean()), 1e-9) * 100, 1
                ),
            })
        return {
            "counts": {
                "benign": int(len(benign)),
                "malignant": int(len(malignant)),
                "total": int(len(df)),
            },
            "features": sorted(stats, key=lambda s: s["diff_pct"], reverse=True),
        }
