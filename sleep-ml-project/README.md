# Sleep Health and Lifestyle Analysis

A machine learning project to analyze sleep patterns and their relationship with lifestyle factors using the Sleep Health and Lifestyle Dataset.

## Dataset

**Source:** [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

The dataset contains information about sleep duration, quality, lifestyle factors, and health metrics.

## Project Objectives

- Analyze factors influencing sleep quality and duration
- Identify patterns and correlations between lifestyle and sleep health
- Build predictive models for sleep-related outcomes
- Provide insights for improving sleep health

## Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone this repository
```bash
git clone <your-repo-url>
cd sleep-health-analysis
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Download the dataset
```python
import kagglehub
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
```

## Project Structure

```
sleep-health-analysis/
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_validation.py  # Data quality checks
│   ├── preprocessing.py    # Data cleaning and transformation
│   └── models.py           # ML model implementations
├── results/                # Model outputs and visualizations
├── README.md
└── requirements.txt
```

## Data Validation Steps

1. Load and inspect data structure
2. Check for missing values and duplicates
3. Validate data ranges and realistic values
4. Analyze distributions and outliers
5. Examine categorical variable consistency
6. Correlation analysis
7. Train/test split preparation

## Usage

```python
# Load and validate data
import pandas as pd
df = pd.read_csv('path/to/Sleep_health_and_lifestyle_dataset.csv')

# Run validation checks
# (Add your validation script here)
```

## Features

Expected features in the dataset:
- Sleep duration
- Sleep quality
- Physical activity level
- Stress levels
- BMI category
- Heart rate
- Daily steps
- Age, gender, occupation

## Models

(To be implemented)
- Regression models for sleep duration prediction
- Classification models for sleep quality categories
- Clustering for lifestyle pattern identification

## Results

(To be updated with findings)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Jason Lin

## Acknowledgments

- Dataset provided by UOM190346A on Kaggle
- [Add any other acknowledgments]