"""Load and preprocess pharmacokinetics / graph datasets."""

from pathlib import Path

import pandas as pd


def get_data_dir() -> Path:
    """Return the project data directory."""
    return Path(__file__).resolve().parents[2] / "data"


def load_raw_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV from data/raw/.

    Args:
        filename: Name of the file under data/raw/.

    Returns:
        DataFrame with the file contents.
    """
    path = get_data_dir() / "raw" / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    print(f"Data directory: {get_data_dir()}")
