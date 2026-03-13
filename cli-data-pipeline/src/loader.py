import csv
from pathlib import Path 

from src.validators import LoadError


def load_csv(filepath: str) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    path = Path(filepath)

    if not path.exists():
        raise LoadError(str(path), "file does not exist")
    #making sure the file is a csv 
    if path.suffix.lower() != ".csv":
        raise LoadError(str(path), f"expected .csv, got {path.suffix}")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except UnicodeDecodeError as e:
        raise LoadError(str(path), f"encoding error: {e}") from e
    except csv.Error as e:
        raise LoadError(str(path), f"CSV parsing error: {e}") from e

if __name__ == "__main__":
    data = load_csv("data/sales_raw.csv")
    print(f"Loaded {len(data)} rows")
    print(f"First row: {data[0]}")

