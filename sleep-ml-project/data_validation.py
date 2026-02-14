"""
Sleep Health Project Dataset Validation - Pydantic Implementation
----------------------------------------------
Industry-standard data validation using Pydantic for schema enforcement.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
from pydantic import BaseModel, Field, validator, ValidationError
from datetime import datetime
from typing import Literal
import json

# =======================================
# PYDANTIC SCHEMA DEFINITION
# =======================================

class SleepHealthRecord(BaseModel):

    person_id: int = Field(
        ...,
        alias='Person ID',
        gt = 0,
        description="Unique identifier for each person"
    )

################################
#Main
################################

# Download latest version
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")

if __name__ == "__main__":
    print("Loading dataset...")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "Sleep_health_and_lifestyle_dataset.csv")

try:
    df = pd.read_csv(csv_file)
    print(f"Dataset loaded: {df.shape[0]}, rows, {df.shape[1]} columns\n")



    # ==== Step 2. Data Quality Checks ====

    #Check for missing values
    print("=" *50)
    print("Missing Value Check")
    print("=" *50)

    #count missing values per column
except FileNotFoundError:
    print(f"Error: File not found at {csv_file}")
    print("Update the csv_file path in the script")
except Exception as e:
    print(f"Error: {e}")
    #need to find out what this does
    import traceback
    traceback.print_exc()
