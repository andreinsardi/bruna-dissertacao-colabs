import unittest

import numpy as np
import pandas as pd

from scripts.definitive_analysis import (
    aggregate_cv_results,
    build_actor_folds,
    confusion_pairs_by_group,
    group_emotion_metrics,
    validate_actor_folds,
)


class DefinitiveAnalysisTests(unittest.TestCase):
    @staticmethod
    def synthetic_manifest(n_actors=30):
        rows = []
        emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
        for actor in range(n_actors):
            race = "African American" if actor % 2 else "Caucasian"
            sex = "Female" if (actor // 2) % 2 else "Male"
            age_bin = ["20-29", "30-39", "40-49", "50-59", "60+"][actor % 5]
            for label, emotion in enumerate(emotions):
                rows.append(
                    {
                        "ActorID": actor,
                        "Race": race,
                        "Sex": sex,
                        "age_bin": age_bin,
                        "label_code": emotion,
                        "y": label,
                    }
                )
        return pd.DataFrame(rows)

    def test_actor_folds_cover_every_sample_without_identity_leakage(self):
        manifest = self.synthetic_manifest()
        folds, assignments, strategy = build_actor_folds(
            manifest,
            n_splits=5,
            seed=42,
            validation_fraction=0.15,
        )

        self.assertEqual(len(folds), 5)
        self.assertEqual(assignments["ActorID"].nunique(), 30)
        self.assertIn(strategy, {"Race__Sex", "Race", "Sex", "unstratified"})
        audit = validate_actor_folds(manifest, folds)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["n_samples"], len(manifest))
        self.assertEqual(audit["test_appearances_min"], 1)
        self.assertEqual(audit["test_appearances_max"], 1)

        for fold in folds:
            train_actors = set(manifest.iloc[fold["train_idx"]]["ActorID"])
            val_actors = set(manifest.iloc[fold["val_idx"]]["ActorID"])
            test_actors = set(manifest.iloc[fold["test_idx"]]["ActorID"])
            self.assertFalse(train_actors & val_actors)
            self.assertFalse(train_actors & test_actors)
            self.assertFalse(val_actors & test_actors)

    def test_cv_aggregation_returns_mean_std_and_confusion_matrices(self):
        records = [
            {
                "model": "EmoT",
                "fold": 1,
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred": np.array([0, 1, 1, 1]),
            },
            {
                "model": "EmoT",
                "fold": 2,
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred": np.array([0, 0, 0, 1]),
            },
        ]

        fold_metrics, summary, confusions = aggregate_cv_results(records, labels=[0, 1])

        self.assertEqual(len(fold_metrics), 2)
        self.assertEqual(summary.iloc[0]["n_folds"], 2)
        self.assertAlmostEqual(summary.iloc[0]["accuracy_mean"], 0.75)
        self.assertEqual(confusions["EmoT"]["mean"].shape, (2, 2))
        self.assertEqual(confusions["EmoT"]["std"].shape, (2, 2))
        self.assertEqual(int(confusions["EmoT"]["pooled_raw"].sum()), 8)

    def test_group_emotion_and_confusion_pair_tables_are_quantitative(self):
        meta = pd.DataFrame(
            {
                "Race": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "Sex": ["F", "F", "M", "M", "F", "F", "M", "M"],
            }
        )
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        emotion = group_emotion_metrics(
            meta,
            y_true,
            y_pred,
            group_attributes=["Race", "Sex"],
            class_names=["happy", "sad"],
            min_support=1,
        )
        pairs = confusion_pairs_by_group(
            meta,
            y_true,
            y_pred,
            group_attributes=["Race"],
            class_names=["happy", "sad"],
            min_true_support=1,
        )

        row = emotion.query("protected_attribute == 'Race' and group == 'A' and emotion == 'happy'").iloc[0]
        self.assertEqual(row["support_true"], 2)
        self.assertAlmostEqual(row["recall"], 0.5)
        self.assertEqual(set(pairs["true_emotion"]), {"happy", "sad"})
        self.assertTrue((pairs["error_rate_within_true"] > 0).all())


if __name__ == "__main__":
    unittest.main()
