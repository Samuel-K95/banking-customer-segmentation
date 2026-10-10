import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandera as pa

# Define Pandera Schema for the Interpretable Matrix
track_b_schema = pa.DataFrameSchema({
    "age": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=True, coerce=True),
    "gross_income": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=True, coerce=True),
    "total_products_owned": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True)
})

def build_product_portfolio_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds product portfolio, returning both interpretable and model matrices."""
    df_current = df.copy()
    
    df_current['age'] = pd.to_numeric(df_current['age'], errors='coerce')
    df_current['seniority_months'] = pd.to_numeric(df_current['seniority_months'], errors='coerce')
    
    if 'province_name' in df_current.columns:
        province_medians = df_current.groupby('province_name')['gross_income'].transform('median')
        df_current['gross_income'] = df_current['gross_income'].fillna(province_medians)
        
    for col in ['gross_income', 'age', 'seniority_months']:
        if col in df_current.columns:
            df_current[col] = df_current[col].fillna(df_current[col].median())
            
    products = [c for c in df_current.columns if c.startswith('has_')]
    df_current['total_products_owned'] = df_current[products].fillna(0).sum(axis=1)

    if 'customer_segment' in df_current.columns:
        df_current = pd.get_dummies(df_current, columns=['customer_segment'], dummy_na=False, dtype=int)
        
    if 'gender' in df_current.columns:
        df_current['is_female'] = np.where(df_current['gender'] == 'V', 1, 0)
        
    binary_text_cols = ['foreigner_index', 'deceased_index', 'spouse_index']
    for col in binary_text_cols:
        if col in df_current.columns:
            df_current[col] = df_current[col].map({'S': 1, 'N': 0, 's': 1, 'n': 0})
            df_current[col] = df_current[col].fillna(0).astype(int)
            
    numeric_flags = ['new_customer_index', 'activity_index']
    for col in numeric_flags:
        if col in df_current.columns:
            df_current[col] = df_current[col].fillna(0).astype(int)
        
    df_current[products] = df_current[products].fillna(0)
    
    cols_to_drop = ['gender', 'province_name', 'snapshot_date', 'customer_type', 'customer_type_primary', 'customer_relation_type', 'address_type']
    df_current = df_current.drop(columns=[c for c in cols_to_drop if c in df_current.columns], errors='ignore')
    
    for col in df_current.columns:
        if df_current[col].dtype == 'object':
            df_current[col] = pd.to_numeric(df_current[col], errors='coerce').fillna(0)
    
    # Validate interpretable matrix strictly (Hard Gate)
    interpretable_df = df_current.copy()
    interpretable_df = track_b_schema.validate(interpretable_df)
        
    # Normalization for Model Matrix
    model_df = interpretable_df.copy()
    continuous_features = ['age', 'gross_income', 'seniority_months', 'total_products_owned']
    continuous_features = [c for c in continuous_features if c in model_df.columns]
    
    scaler = StandardScaler()
    if continuous_features:
        model_df[continuous_features] = scaler.fit_transform(model_df[continuous_features])
        
    return interpretable_df, model_df

if __name__ == "__main__":
    from src.customer_segmentation.data.santander_dataset_loader import load_santander_raw_data
    
    print("Loading raw Santander dataset (with chunking and target date filtering)...")
    raw_df = load_santander_raw_data("data/raw/santander/train_ver2.csv", target_date="2016-05-28")
    
    print("Building features...")
    interpretable_df, model_df = build_product_portfolio_features(raw_df)
    
    os.makedirs("data/processed", exist_ok=True)
    
    interpretable_df.to_csv("data/processed/track_b_features_interpretable.csv", index=False)
    model_df.to_csv("data/processed/track_b_features_model.csv", index=False)
    print("Saved interpretable and model matrices to data/processed/")
