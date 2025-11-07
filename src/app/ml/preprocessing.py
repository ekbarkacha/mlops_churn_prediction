"""
Module: ml/preprocessing.py
============================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Centralized preprocessing utilities for inference data.
Eliminates code duplication by providing a single source of truth
for data cleaning, encoding, feature engineering, and scaling.
"""
import os
import joblib
import pandas as pd
from typing import Optional
from src.utils.const import scaler_file_name, label_encoders_file_name


class InferencePreprocessor:
    """
    Centralized preprocessor for inference data.
    Applies the same transformations as during training.
    """

    def __init__(self, version_dir: str):
        """
        Initialize preprocessor by loading artifacts from model version directory.

        Args:
            version_dir: Path to model version directory containing preprocessors.
        """
        self.preprocessors_path = os.path.join(version_dir, "preprocessors")
        self.scaler = joblib.load(os.path.join(self.preprocessors_path, scaler_file_name))
        self.encoders = joblib.load(os.path.join(self.preprocessors_path, label_encoders_file_name))

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and type conversions.

        Args:
            df: Input dataframe to clean.

        Returns:
            Cleaned dataframe.
        """
        df = df.copy(deep=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using stored label encoders.

        Args:
            df: Input dataframe with categorical columns.

        Returns:
            Dataframe with encoded categorical columns.
        """
        df = df.copy(deep=True)
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
        return df

    def feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features that are not used in the model.

        Args:
            df: Input dataframe.

        Returns:
            Dataframe with selected features only.
        """
        drop_features = [
            'customerID',
            'PhoneService',
            'gender',
            'StreamingTV',
            'StreamingMovies',
            'MultipleLines',
            'InternetService'
        ]
        df = df.copy(deep=True)
        df.drop(columns=drop_features, inplace=True, errors='ignore')
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features using stored scaler.

        Args:
            df: Input dataframe with numeric columns.

        Returns:
            Dataframe with scaled numeric features.
        """
        df = df.copy(deep=True)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[num_cols] = self.scaler.transform(df[num_cols])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline:
        1. Data cleaning
        2. Categorical encoding
        3. Feature selection
        4. Feature scaling

        Args:
            df: Raw input dataframe.

        Returns:
            Fully preprocessed dataframe ready for model inference.
        """
        df = self.data_cleaning(df)
        df = self.encode_categorical(df)
        df = self.feature_selection(df)
        df = self.scale_features(df)
        return df
