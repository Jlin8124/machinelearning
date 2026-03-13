# CLI Data Pipeline

A command-line tool to clean, validate, and summarize messy sales CSV data.

## Project Structure

```
cli-data-pipeline/
├── data/
│   └── sales_raw.csv       ← messy input data
├── src/
│   ├── __init__.py
│   ├── cli.py              ← argparse entry point
│   ├── loader.py           ← reads CSV files
│   ├── cleaners.py         ← cleaning functions
│   ├── validators.py       ← validation + custom exceptions
│   └── stats.py            ← summary statistics
├── tests/
│   └── test_cleaners.py
└── README.md
```

## Usage

```bash
# Run the pipeline (text output)
python -m src.cli data/sales_raw.csv

# Show invalid rows
python -m src.cli data/sales_raw.csv --show-invalid

# Stats only
python -m src.cli data/sales_raw.csv --stats-only

# JSON output
python -m src.cli data/sales_raw.csv --output json
```

## Running Tests

```bash
pytest tests/
```
