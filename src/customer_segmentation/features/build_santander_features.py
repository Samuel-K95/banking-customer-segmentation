import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def build_product_portfolio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the binary product portfolio matrix and scales demographics for the Santander dataset."""
    df_current = df.copy()
    
    # 1. Numeric conversions
    df_current['age'] = pd.to_numeric(df_current['age'], errors='coerce')
    df_current['seniority_months'] = pd.to_numeric(df_current['seniority_months'], errors='coerce')
    
    # 2. Impute Income (gross_income) by Province (province_name) median
    if 'province_name' in df_current.columns:
        province_medians = df_current.groupby('province_name')['gross_income'].transform('median')
        df_current['gross_income'] = df_current['gross_income'].fillna(province_medians)
        
    # Fill remaining demographics with overall median
    for col in ['gross_income', 'age', 'seniority_months']:
        if col in df_current.columns:
            df_current[col] = df_current[col].fillna(df_current[col].median())
            
    # 3. Categorical Encoding & Product Vectorization
    products = [c for c in df_current.columns if c.startswith('has_')]
    
    if 'customer_segment' in df_current.columns:
        df_current = pd.get_dummies(df_current, columns=['customer_segment'], dummy_na=False, dtype=int)
        
    if 'gender' in df_current.columns:
        df_current['is_female'] = np.where(df_current['gender'] == 'V', 1, 0)
        
    # Fill missing products with 0
    df_current[products] = df_current[products].fillna(0)
    
    # Drop unneeded text columns
    df_current = df_current.drop(columns=['gender', 'province_name', 'snapshot_date'], errors='ignore')
    
    # 4. Normalization
    continuous_features = ['age', 'gross_income', 'seniority_months']
    continuous_features = [c for c in continuous_features if c in df_current.columns]
    
    scaler = StandardScaler()
    if continuous_features:
        df_current[continuous_features] = scaler.fit_transform(df_current[continuous_features])
        
    return df_current

if __name__ == "__main__":
    from src.customer_segmentation.data.santander_dataset_loader import load_santander_raw_data
    
    print("Loading raw Santander dataset (with chunking and target date filtering)...")
    raw_df = load_santander_raw_data("data/raw/santander/train_ver2.csv", target_date="2016-05-28")
    
    print("Building features...")
    features_df = build_product_portfolio_features(raw_df)
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/track_b_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
