from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data.csv"

FEATURE_DESCRIPTIONS = {
    "radius_mean": "Average radius of the cell nuclei.",
    "texture_mean": "Standard deviation of gray-scale values in the cell image.",
    "perimeter_mean": "Average perimeter of the cell nuclei.",
    "area_mean": "Average area covered by the cell nuclei.",
    "smoothness_mean": "Local variation in radius lengths.",
    "compactness_mean": "Perimeter squared divided by area minus one.",
    "concavity_mean": "Severity of concave portions of the cell contour.",
    "concave points_mean": "Number of concave portions of the cell contour.",
    "symmetry_mean": "Symmetry score of the cell nuclei.",
    "fractal_dimension_mean": "Coastline-approximation style complexity of the boundary.",
    "radius_se": "Standard error for nucleus radius.",
    "texture_se": "Standard error for texture.",
    "perimeter_se": "Standard error for perimeter.",
    "area_se": "Standard error for area.",
    "smoothness_se": "Standard error for smoothness.",
    "compactness_se": "Standard error for compactness.",
    "concavity_se": "Standard error for concavity.",
    "concave points_se": "Standard error for concave points.",
    "symmetry_se": "Standard error for symmetry.",
    "fractal_dimension_se": "Standard error for fractal dimension.",
    "radius_worst": "Largest observed mean radius among the most severe nuclei.",
    "texture_worst": "Largest observed texture score among the most severe nuclei.",
    "perimeter_worst": "Largest observed perimeter among the most severe nuclei.",
    "area_worst": "Largest observed area among the most severe nuclei.",
    "smoothness_worst": "Largest observed smoothness score among the most severe nuclei.",
    "compactness_worst": "Largest observed compactness score among the most severe nuclei.",
    "concavity_worst": "Largest observed concavity score among the most severe nuclei.",
    "concave points_worst": "Largest observed concave-point score among severe nuclei.",
    "symmetry_worst": "Largest observed symmetry score among the most severe nuclei.",
    "fractal_dimension_worst": "Largest observed fractal-dimension score among severe nuclei.",
}


class BreastCancerData:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.df = self._load()
        self.feature_names = [
            column for column in self.df.columns if column not in {"id", "diagnosis", "target"}
        ]

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df = df.drop(columns=["Unnamed: 32"], errors="ignore")
        df["target"] = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
        return df

    def features_frame(self) -> pd.DataFrame:
        return self.df[self.feature_names].copy()

    def target_series(self) -> pd.Series:
        return self.df["target"].copy()

    def metadata(self) -> dict:
        features = []
        for name in self.feature_names:
            series = self.df[name]
            features.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "description": FEATURE_DESCRIPTIONS.get(name, "Numeric cell-nucleus measurement."),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "mean": float(series.mean()),
                }
            )

        counts = self.df["diagnosis"].value_counts().to_dict()
        return {
            "features": features,
            "feature_names": self.feature_names,
            "class_counts": {
                "benign": int(counts.get("B", 0)),
                "malignant": int(counts.get("M", 0)),
            },
            "rows": int(len(self.df)),
        }

    def random_sample(self) -> dict:
        row = self.df.sample(1).iloc[0]
        features = {name: float(row[name]) for name in self.feature_names}
        return {
            "id": int(row["id"]),
            "diagnosis": row["diagnosis"],
            "label": "Malignant" if row["diagnosis"] == "M" else "Benign",
            "features": features,
        }

    def median_features(self) -> dict[str, float]:
        return {name: float(self.df[name].median()) for name in self.feature_names}
