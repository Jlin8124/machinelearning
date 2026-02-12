import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "Sleep_health_and_lifestyle_dataset.csv")
df = pd.read_csv(csv_file)

print("Dataset Shape:", df.shape)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

#Step 4: Display columnn names and their data types
print("Column Names and Data Types:")
print(df.dtypes)
print("\n")

print("First 5 Rows:")
print(df.head())
print("\n")

print("Dataset Info")
df.info()
print("\n")

print("Statistical Summary")
print(df.describe())

#Outlier detection (IQR method)
Q1 = df.select_dtypes(include='number').quantile(0.25)
Q3 = df.select_dtypes(include='number').quantile(0.75)
IQR = Q3 - Q1

numeric_df = df.select_dtypes(include='number')

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).sum()

print("Outlier counts per columne :\n", outliers)
