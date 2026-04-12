"""
Data Loader
============
Utilities for loading TSP instances from the Kaggle TSPLIB dataset,
the bundled sample CSV, or generating random city configurations.

Data source priority:
    1. Kaggle TSPLIB dataset (primary — requires kagglehub + credentials)
    2. Bundled sample CSV (fallback for demo/testing)
    3. Random city generator (always available)
"""

import os
import ast
import numpy as np
import pandas as pd


# Relative path to the bundled sample CSV
SAMPLE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "sample", "sample_tsp.csv"
)


def load_kaggle_dataset() -> pd.DataFrame | None:
    """
    Download and load the Kaggle TSPLIB dataset using kagglehub.

    Attempts to download the dataset from Kaggle. If kagglehub is not
    installed or Kaggle credentials are not configured, returns None.

    Returns:
        DataFrame with the TSPLIB dataset, or None if download fails.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "ziya07/traveling-salesman-problem-tsplib-dataset"
        )
        # Find the CSV file in the downloaded directory
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            print(f"[WARNING] No CSV files found in: {path}")
            return None

        csv_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(csv_path)
        return df

    except ImportError:
        print("[INFO] kagglehub not installed. Using fallback data source.")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to download Kaggle dataset: {e}")
        return None


def load_sample_dataset() -> pd.DataFrame | None:
    """
    Load the bundled sample CSV file included in the repository.
    This serves as a reliable fallback when Kaggle is not available.

    Returns:
        DataFrame with sample TSP instances, or None if file not found.
    """
    if os.path.exists(SAMPLE_CSV_PATH):
        return pd.read_csv(SAMPLE_CSV_PATH)
    return None


def load_dataset() -> tuple[pd.DataFrame | None, str]:
    """
    Load TSP data with automatic fallback strategy.

    Tries sources in order:
        1. Kaggle TSPLIB dataset (primary)
        2. Bundled sample CSV (fallback)

    Returns:
        Tuple of (DataFrame or None, source_name string).
    """
    # Try Kaggle first (primary source)
    df = load_kaggle_dataset()
    if df is not None and len(df) > 0:
        return df, "Kaggle TSPLIB Dataset"

    # Fallback to bundled sample
    df = load_sample_dataset()
    if df is not None and len(df) > 0:
        return df, "Bundled Sample CSV"

    return None, "None"


def parse_instance(row: pd.Series, n_cities: int = 20) -> np.ndarray:
    """
    Extract city coordinates from a single dataset row.

    Handles either the Kaggle format (where 'city_coordinates' is a string
    representation of a list of lists) or the fallback format with separate
    columns (City_1_X, City_1_Y...).

    Args:
        row: A single row from the TSPLIB DataFrame.
        n_cities: Number of cities per instance (default 20).

    Returns:
        NumPy array of shape (n_cities, 2) with (x, y) coordinates.
    """
    if "city_coordinates" in row.index:
        coords_str = row["city_coordinates"]
        if isinstance(coords_str, str):
            coords = ast.literal_eval(coords_str)
        else:
            coords = coords_str
        return np.array(coords, dtype=np.float64)

    coords = []
    for i in range(1, n_cities + 1):
        x_col = f"City_{i}_X"
        y_col = f"City_{i}_Y"

        if x_col in row.index and y_col in row.index:
            coords.append([row[x_col], row[y_col]])
        else:
            # Try alternative column naming: City_X_1, City_Y_1
            alt_x = f"City_X_{i}"
            alt_y = f"City_Y_{i}"
            if alt_x in row.index and alt_y in row.index:
                coords.append([row[alt_x], row[alt_y]])
            else:
                raise ValueError(
                    f"Cannot find coordinates for city {i}. "
                    f"Expected columns: {x_col}/{y_col} or {alt_x}/{alt_y}"
                )

    return np.array(coords, dtype=np.float64)


def get_instance_ids(df: pd.DataFrame) -> list:
    """
    Get list of available Instance IDs from the dataset.

    Args:
        df: TSPLIB DataFrame.

    Returns:
        List of Instance_ID values, or range indices if column not found.
    """
    if "instance_id" in df.columns:
        return sorted(df["instance_id"].unique().tolist())
    if "Instance_ID" in df.columns:
        return sorted(df["Instance_ID"].unique().tolist())
    return list(range(len(df)))


def detect_n_cities(df: pd.DataFrame) -> int:
    """
    Auto-detect the number of cities from the dataset columns.

    Scans column names for City_N_X patterns and returns the maximum N found.
    If 'num_cities' is present, uses its value.

    Args:
        df: TSPLIB DataFrame.

    Returns:
        Detected number of cities.
    """
    if "num_cities" in df.columns:
        return int(df["num_cities"].max())
        
    max_city = 0
    for col in df.columns:
        if col.startswith("City_") and col.endswith("_X"):
            parts = col.split("_")
            try:
                city_num = int(parts[1])
                max_city = max(max_city, city_num)
            except ValueError:
                continue
    return max_city if max_city > 0 else 20


def get_instance_by_id(df: pd.DataFrame, instance_id) -> pd.Series:
    """
    Retrieve a specific instance row by its ID.

    Args:
        df: TSPLIB DataFrame.
        instance_id: The Instance_ID to look up.

    Returns:
        The matching row as a pandas Series.
    """
    if "instance_id" in df.columns:
        row = df[df["instance_id"] == instance_id].iloc[0]
    elif "Instance_ID" in df.columns:
        row = df[df["Instance_ID"] == instance_id].iloc[0]
    else:
        row = df.iloc[instance_id]
    return row
