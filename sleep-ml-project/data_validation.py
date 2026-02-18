"""
Sleep Health Project Dataset Validation - Pydantic Implementation
----------------------------------------------
Jason Lin

Industry-standard data validation using Pydantic for schema enforcement.

This project analyzes the Sleep Health and Lifestyle Dataset to predict
sleep disorders (None, Insomnia, Sleep Apnea) based on lifestyle and
health metrics. It compares multiple ML models and provides interpretable
feature importance analysis.

Dataset: Sleep Health and Lifestyle Dataset (374 records, 13 features)
Target: Sleep Disorder classification (None / Insomnia / Sleep Apnea)

Models Evaluated:
    - Logistic Regression (baseline)
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Support Vector Machine (SVM)

"""
# ============================================================================
# 1. IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "Sleep_health_and_lifestyle_dataset.csv")

# If data doesn't exist locally, download from Kaggle
if not os.path.exists(CSV_FILE):
    import kagglehub
    path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.rename(
        os.path.join(path, "Sleep_health_and_lifestyle_dataset.csv"),
        CSV_FILE
    )
    print(f"Downloaded dataset to {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# 3. DATA LOADING & EXPLORATION
# ============================================================================

def load_and_explore(path: str) -> pd.DataFrame:
    """Load dataset and print exploratory summary."""
    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes.to_string()}")
    print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
    print(f"\nTarget Distribution (Sleep Disorder):")
    print(f"  - NaN (No Disorder): {df['Sleep Disorder'].isna().sum()}")
    print(f"  - {df['Sleep Disorder'].value_counts().to_string()}")
    print(f"\nNumeric Summary:\n{df.describe().round(2).to_string()}")

    return df
# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """
    Clean and engineer features from raw data.

    Steps:
        1. Fill NaN Sleep Disorder → 'None'
        2. Merge 'Normal Weight' into 'Normal' (BMI Category)
        3. Split Blood Pressure into systolic/diastolic
        4. Encode categorical features
        5. Drop unnecessary columns (Person ID)

    Returns:
        - Processed DataFrame (features only)
        - Target array (encoded)
        - Target class names
    """
    df = df.copy()

    # --- Target variable ---
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    # --- BMI Category cleanup ---
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

    # --- Blood Pressure → systolic + diastolic ---
    bp_split = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df['BP_Systolic'] = bp_split[0]
    df['BP_Diastolic'] = bp_split[1]
    df.drop(columns=['Blood Pressure'], inplace=True)

    # --- Encode target ---
    
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['Sleep Disorder'])
    class_names = list(target_encoder.classes_)

    # --- Encode categorical features ---
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    occupation_encoder = LabelEncoder()
    df['Occupation'] = occupation_encoder.fit_transform(df['Occupation'])
    bmi_encoder = LabelEncoder()
    df['BMI Category'] = bmi_encoder.fit_transform(df['BMI Category'])

    # --- Drop non-feature columns ---
    df.drop(columns=['Person ID', 'Sleep Disorder'], inplace=True)

    encoders = {
        'target': target_encoder,
        'Gender': gender_encoder,
        'Occupation': occupation_encoder,
        'BMI': bmi_encoder
     }

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Features: {list(df.columns)}")
    print(f"Target classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    return df, y, class_names, encoders

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the full ML pipeline."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SLEEP DISORDER PREDICTION — ML PIPELINE                ║")
    print("╚" + "═" * 58 + "╝\n")

    load_and_explore(CSV_FILE)




if __name__ == "__main__":
     main()