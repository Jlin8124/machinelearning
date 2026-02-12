import pandas as pd
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