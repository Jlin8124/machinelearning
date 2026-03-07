"""
Titanic Dataset — Data Cleaning Script
=======================================
A reusable script that takes the raw Titanic CSV and outputs a clean,
model-ready dataset.

Usage:
    python clean_titanic.py

Output:
    titanic_cleaned.csv
"""

import pandas as pd
import numpy as np


def load_data(path: str = None) -> pd.DataFrame:
    """Load Titanic dataset from local path or public URL."""
    if path:
        return pd.read_csv(path)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("Missing values:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"  {col:15s} → {count:4d} ({pct:.1f}%)")
    else:
        print("No missing values.")
    print()


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are irrelevant or too sparse."""
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"Dropped columns: {cols_to_drop}")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with sensible strategies."""
    # Age: median grouped by passenger class
    df["Age"] = df.groupby("Pclass")["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    print("Filled Age with median per Pclass")

    # Embarked: mode (most common value)
    mode_val = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(mode_val)
    print(f"Filled Embarked with mode: {mode_val}")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric."""
    # Sex: binary label encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("Encoded Sex → 0 (male), 1 (female)")

    # Embarked: one-hot encoding (drop first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    print("One-hot encoded Embarked (dropped first category)")

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme outliers in Fare using winsorization."""
    fare_cap = df["Fare"].quantile(0.99)
    n_capped = (df["Fare"] > fare_cap).sum()
    df["Fare"] = df["Fare"].clip(upper=fare_cap)
    print(f"Capped {n_capped} Fare values at 99th percentile (${fare_cap:.2f})")
    return df


def clean_titanic(input_path: str = None, output_path: str = "titanic_cleaned.csv") -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Parameters
    ----------
    input_path : str, optional
        Path to raw CSV. If None, downloads from public URL.
    output_path : str
        Where to save the cleaned CSV.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    print("🚢 Titanic Data Cleaning Pipeline")
    print("=" * 50)
    print()

    # Load
    df = load_data(input_path)
    inspect_data(df)

    # Clean
    print("CLEANING STEPS")
    print("-" * 40)
    df = drop_columns(df)
    df = fill_missing(df)
    df = encode_categoricals(df)
    df = handle_outliers(df)
    print()

    # Verify
    remaining_missing = df.isnull().sum().sum()
    print("FINAL CHECK")
    print("-" * 40)
    print(f"Shape:          {df.shape}")
    print(f"Missing values: {remaining_missing}")
    print(f"Columns:        {list(df.columns)}")
    print()

    # Export
    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned dataset → {output_path}")

    return df


if __name__ == "__main__":
    clean_titanic()