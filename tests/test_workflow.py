import unittest
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
LEAKAGE_COLUMNS = {"G1", "G2", "G3"}
RANDOM_STATE = 42


class StudentRiskWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA_PATH, sep=";")
        cls.data["at_risk"] = (cls.data["G3"] < 10).astype(int)
        cls.features = cls.data.drop(columns=[*LEAKAGE_COLUMNS, "at_risk"])
        cls.target = cls.data["at_risk"]

    def test_dataset_integrity_and_target_distribution(self):
        self.assertEqual(self.data.shape, (395, 34))
        self.assertEqual(int(self.data.isna().sum().sum()), 0)
        self.assertEqual(self.data["at_risk"].value_counts().to_dict(), {0: 265, 1: 130})

    def test_model_features_exclude_grade_leakage(self):
        self.assertTrue(LEAKAGE_COLUMNS.isdisjoint(self.features.columns))
        self.assertEqual(self.features.shape[1], 30)

    def test_pipeline_trains_and_predicts_feature_only_rows(self):
        numeric_features = self.features.select_dtypes(include="number").columns.tolist()
        categorical_features = self.features.select_dtypes(exclude="number").columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=25,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        features_train, features_test, target_train, target_test = train_test_split(
            self.features,
            self.target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=self.target,
        )

        pipeline.fit(features_train, target_train)
        predictions = pipeline.predict(features_test)

        self.assertEqual(len(predictions), len(target_test))
        self.assertEqual(set(predictions), {0, 1})


if __name__ == "__main__":
    unittest.main()
