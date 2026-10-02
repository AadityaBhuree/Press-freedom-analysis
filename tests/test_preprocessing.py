"""Tests for data preprocessing pipeline and target leakage prevention."""

import os
import json
import pandas as pd
import pytest

from preprocessing import run_preprocessing


LEAKAGE_COLUMNS = [
    "Global Score",
    "Position 2022",
    "Position 2021",
    "Position_Change",
    "Situation",
    "Country",
    "ISO Code",
    "target",
]

EXPECTED_FEATURES = [
    "Region",
    "Politic Score",
    "Economic Score",
    "Legislative Score",
    "Social Score",
    "Security Score",
    "Journalist Killed",
    "Media Workers Killed",
    "Journalist Imprisoned",
    "Media Workers Imprisoned",
    "Press_Danger_Index",
    "Score_Variance",
    "Score_Range",
    "Score_Min",
]


class TestPreprocessingPipeline:
    """Test suite ensuring data integrity and zero label leakage."""

    @pytest.fixture(scope="module", autouse=True)
    def setup_data(self):
        """Run preprocessing once for test assertions."""
        run_preprocessing(data_path="dataset.csv", output_dir=".")

    def test_processed_files_exist(self):
        """Verify that all processed datasets and encoder artifacts are generated."""
        assert os.path.exists("X_train.csv")
        assert os.path.exists("X_test.csv")
        assert os.path.exists("y_train.csv")
        assert os.path.exists("y_test.csv")
        assert os.path.exists(os.path.join("models", "scaler.joblib"))
        assert os.path.exists(os.path.join("models", "region_encoder.joblib"))
        assert os.path.exists(os.path.join("models", "label_mapping.json"))
        assert os.path.exists(os.path.join("models", "feature_names.json"))

    def test_no_target_leakage_in_features(self):
        """CRITICAL: Ensure target-defining columns are strictly excluded from feature sets."""
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")

        for col in LEAKAGE_COLUMNS:
            assert col not in X_train.columns, f"Target leakage detected! '{col}' is in X_train"
            assert col not in X_test.columns, f"Target leakage detected! '{col}' is in X_test"

    def test_feature_list_matches_expected(self):
        """Verify feature vector matches expected exogenous indicator set."""
        X_train = pd.read_csv("X_train.csv")
        with open(os.path.join("models", "feature_names.json"), "r") as f:
            saved_features = json.load(f)

        assert list(X_train.columns) == EXPECTED_FEATURES
        assert saved_features == EXPECTED_FEATURES
        assert len(X_train.columns) == 14

    def test_no_nan_values(self):
        """Ensure clean preprocessing with zero nulls."""
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv")
        y_test = pd.read_csv("y_test.csv")

        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()
        assert not y_train.isnull().any().any()
        assert not y_test.isnull().any().any()

    def test_target_values_valid_range(self):
        """Ensure ordinal target labels map strictly to 0..4."""
        y_train = pd.read_csv("y_train.csv").values.ravel()
        y_test = pd.read_csv("y_test.csv").values.ravel()

        valid_labels = {0, 1, 2, 3, 4}
        assert set(y_train).issubset(valid_labels)
        assert set(y_test).issubset(valid_labels)

    def test_train_test_split_ratio(self):
        """Ensure 80/20 train/test split consistency."""
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")

        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.18 <= test_ratio <= 0.22
