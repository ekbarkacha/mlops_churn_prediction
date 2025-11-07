"""
Module: utils/csv_utils.py
============================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
CSV file utilities for safe reading and file operations.
"""
import os
import pandas as pd


def read_csv_safe(path: str) -> pd.DataFrame:
    """
    Safely read a CSV file. Returns empty dataframe if file doesn't exist or is empty.

    Args:
        path: Path to CSV file.

    Returns:
        pd.DataFrame: Dataframe read from CSV or empty dataframe.
    """
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def file_exists(path: str) -> bool:
    """
    Check if file exists.

    Args:
        path: File path.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.exists(path)
