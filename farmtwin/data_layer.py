"""
FarmTwin v2 — Data Layer
Handles all data loading, preprocessing, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING MODULES
# ═══════════════════════════════════════════════════════════════════

def load_full_dataset(path='data/FarmTwin_Dataset_v2.csv'):
    """Load the complete FarmTwin dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_weather_data(df):
    """Extract weather-related columns."""
    cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct']
    return df[cols].copy()

def load_soil_data(df):
    """Extract soil-related columns."""
    cols = ['Soil_Type', 'Soil_Moisture_pct']
    return df[cols].copy()

def load_crop_data(df):
    """Extract crop and context columns."""
    cols = ['Crop_Type', 'Season', 'Year', 'Location']
    return df[cols].copy()

def load_management_data(df):
    """Extract farm management (controllable) columns."""
    cols = ['Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']
    return df[cols].copy()


# ═══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def clean_data(df):
    """Handle missing values and clip out-of-range data."""
    # Fill numeric NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaN with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip negative values for physical quantities
    non_negative = ['Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
                    'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']
    for col in non_negative:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    print(f"✅ Data cleaned. Remaining NaN: {df.isnull().sum().sum()}")
    return df


def normalize_data(df, numeric_cols=None):
    """Normalize numeric features using StandardScaler."""
    if numeric_cols is None:
        numeric_cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct',
                        'Soil_Moisture_pct', 'Irrigation_mm',
                        'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler


def feature_engineering(df):
    """Create additional useful features."""
    df = df.copy()

    # Total water available to crop
    df['Total_Water'] = df['Rainfall_mm'] + df['Irrigation_mm']

    # Total NPK
    df['Total_NPK'] = df['N_Fertilizer'] + df['P_Fertilizer'] + df['K_Fertilizer']

    # N ratio (nitrogen dominance)
    df['N_Ratio'] = df['N_Fertilizer'] / (df['Total_NPK'] + 1)

    print(f"✅ Feature engineering done. New columns: Total_Water, Total_NPK, N_Ratio")
    return df


def encode_categoricals(df, categorical_cols=None):
    """One-hot encode categorical features."""
    if categorical_cols is None:
        categorical_cols = ['Crop_Type', 'Soil_Type', 'Season', 'Location']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df_out = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    print(f"✅ Encoded {len(categorical_cols)} categorical columns → {encoded_df.shape[1]} features")
    return df_out, encoder


# ═══════════════════════════════════════════════════════════════════
# 3. TIME-BASED SPLIT (🔥 Key differentiator per Paper 8)
# ═══════════════════════════════════════════════════════════════════

def time_based_split(df, split_year=2022):
    """
    Split data by year — NOT random!
    Train on past data, test on future data.
    This ensures the model is evaluated on its ability to predict the future.
    """
    train = df[df['Year'] < split_year].copy()
    test = df[df['Year'] >= split_year].copy()
    print(f"✅ Time-based split at year {split_year}")
    print(f"   Train: {len(train)} rows (years < {split_year})")
    print(f"   Test:  {len(test)} rows (years ≥ {split_year})")
    return train, test


# ═══════════════════════════════════════════════════════════════════
# 4. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def prepare_data(path='data/FarmTwin_Dataset_v2.csv', split_year=2022):
    """
    Complete data pipeline:
    load → clean → feature engineer → encode → time-split
    Returns X_train, X_test, y_train, y_test, encoder, scaler
    """
    # Load
    df = load_full_dataset(path)

    # Clean
    df = clean_data(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Time-based split BEFORE encoding (to prevent data leakage)
    train_df, test_df = time_based_split(df, split_year)

    # Target
    y_train = train_df['Yield_kg_per_ha']
    y_test = test_df['Yield_kg_per_ha']

    # Drop target + Year (not a predictive feature)
    drop_cols = ['Yield_kg_per_ha', 'Year']
    X_train_raw = train_df.drop(columns=drop_cols)
    X_test_raw = test_df.drop(columns=drop_cols)

    # Encode categoricals
    categorical_cols = ['Crop_Type', 'Soil_Type', 'Season', 'Location']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train_raw[categorical_cols])

    def encode_split(X, enc):
        encoded = enc.transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(categorical_cols), index=X.index)
        return pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

    X_train = encode_split(X_train_raw, encoder)
    X_test = encode_split(X_test_raw, encoder)

    # Normalize
    scaler = StandardScaler()
    numeric_cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
                    'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer',
                    'Total_Water', 'Total_NPK', 'N_Ratio']
    scaler.fit(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"\n🎯 Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test, encoder, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, enc, scl = prepare_data()
    print("\n✅ Data Layer ready!")
    print(f"Features: {list(X_train.columns)}")
