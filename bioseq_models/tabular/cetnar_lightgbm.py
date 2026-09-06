"""Cetnar et al. feature-based mRNA stability model bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..adapters.common import CHECKPOINT_DIR


TIMEPOINTS = ("T0", "T2", "T4", "T8", "T16")


class CetnarLightGBM:
    """Six-model LightGBM cascade operating on the paper's feature table.

    This model requires five isoforms' RppH, structure, translation, and
    categorical nucleotide features. It cannot infer those fields from a bare
    nucleotide string, so it is deliberately separate from ``load_model``.
    """

    def __init__(self, timepoint_models, decay_model) -> None:
        self.timepoint_models = dict(timepoint_models)
        self.decay_model = decay_model

    @classmethod
    def from_checkpoint_dir(
        cls, checkpoint_dir: str | Path = CHECKPOINT_DIR
    ) -> "CetnarLightGBM":
        import lightgbm as lgb

        root = Path(checkpoint_dir)
        timepoint_models = {
            timepoint: lgb.Booster(
                model_file=str(root / f"cetnar_lightgbm_{timepoint}.txt")
            )
            for timepoint in TIMEPOINTS
        }
        decay_model = lgb.Booster(
            model_file=str(root / "cetnar_lightgbm_decay_rate.txt")
        )
        return cls(timepoint_models, decay_model)

    @staticmethod
    def _select_features(frame: pd.DataFrame, model) -> pd.DataFrame:
        missing = [name for name in model.feature_name() if name not in frame]
        if missing:
            raise ValueError(
                f"Cetnar feature table is missing {len(missing)} columns; "
                f"first missing column: {missing[0]!r}"
            )
        selected = frame.loc[:, model.feature_name()].copy()
        categorical_names = [
            name
            for name in model.feature_name()
            if name.startswith(("rppH_", "GQUAD_", "iMOTIF_", "CsrA_", "nt_"))
        ]
        categories = model.pandas_categorical or []
        if len(categorical_names) != len(categories):
            raise ValueError("Cetnar checkpoint categorical metadata is inconsistent")
        for column, vocabulary in zip(categorical_names, categories):
            selected[column] = pd.Categorical(selected[column], categories=vocabulary)
        return selected

    def predict_timepoints(self, features: pd.DataFrame) -> pd.DataFrame:
        predictions = {}
        for timepoint, model in self.timepoint_models.items():
            selected = self._select_features(features, model)
            predictions[f"predicted_{timepoint}"] = model.predict(selected)
        return pd.DataFrame(predictions, index=features.index)

    def predict(self, features: pd.DataFrame):
        augmented = features.copy()
        predictions = self.predict_timepoints(augmented)
        for column in predictions:
            augmented[column] = predictions[column]
        selected = self._select_features(augmented, self.decay_model)
        return self.decay_model.predict(selected)
