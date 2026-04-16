"""
FarmTwin v2 — Data Integration Layer
Merges multiple data sources into a unified dataset.
"""
import pandas as pd
import os


def merge_data(weather_df=None, soil_df=None, crop_df=None, management_df=None):
    """
    Merge data from multiple sources into a single DataFrame.
    All DataFrames must have the same number of rows (aligned by index).
    """
    dfs = [df for df in [weather_df, soil_df, crop_df, management_df] if df is not None]
    if not dfs:
        raise ValueError("At least one data source must be provided")

    merged = pd.concat(dfs, axis=1)
    print(f"✅ Merged {len(dfs)} data sources → {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def load_and_merge_csv_files(data_dir='data'):
    """
    Load all CSV files from the data directory and display info.
    Useful for exploring available datasets.
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    datasets = {}

    for f in csv_files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        datasets[f] = df
        print(f"📂 {f}: {df.shape[0]} rows × {df.shape[1]} cols")

    return datasets
