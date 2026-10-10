import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds RFM (Recency, Frequency, Monetary) features from the Indian Bank transactions."""
    
    # 1. Drop NAs
    df = df.dropna().copy()
    
    # 2. Demographic Engineering (Century Pivot & Age calculation)
    try:
        df['DOB'] = pd.to_datetime(df['CustomerDOB'], format='%d/%m/%y', errors='coerce')
        # Pandas parses e.g., '73' as 2073. If year > 2005, it belongs to the 1900s.
        df.loc[df['DOB'].dt.year > 2005, 'DOB'] -= pd.DateOffset(years=100)
        # Calculate Age relative to 2016 (when transactions occurred)
        df['Age'] = 2016 - df['DOB'].dt.year
        df['Age'] = np.where((df['Age'] < 18) | (df['Age'] > 100), np.nan, df['Age'])
    except Exception as e:
        pass

    # One-Hot Encode Gender
    if 'CustGender' in df.columns:
        df = pd.get_dummies(df, columns=['CustGender'], drop_first=True, dtype=int)

    # Parse Transaction Date
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y', errors='coerce')
    
    # 3. Customer-Level Aggregation (RFM)
    max_date = df['TransactionDate'].max()
    
    # Calculate Spend-to-Balance Ratio (add 1 to avoid div by zero)
    df['Spend_Balance_Ratio'] = df['TransactionAmount (INR)'] / (df['CustAccountBalance'] + 1)
    
    # Check if Gender_M column exists
    gender_col = 'CustGender_M' if 'CustGender_M' in df.columns else None
    
    agg_dict = {
        'Age': ('Age', 'first'),
        'AccountBalance': ('CustAccountBalance', 'mean'),
        'Last_Transaction': ('TransactionDate', 'max'),
        'Transaction_Count': ('TransactionID', 'count'),
        'Avg_Transaction_Amt': ('TransactionAmount (INR)', 'mean'),
        'Total_Spend': ('TransactionAmount (INR)', 'sum'),
        'Avg_Spend_Ratio': ('Spend_Balance_Ratio', 'mean')
    }
    
    if gender_col:
        agg_dict['Gender_M'] = (gender_col, 'first')

    customer_df = df.groupby('CustomerID').agg(**agg_dict).dropna()
    
    customer_df['Recency_Days'] = (max_date - customer_df['Last_Transaction']).dt.days
    
    # 4. Normalization
    features = ['Age', 'AccountBalance', 'Transaction_Count', 'Avg_Transaction_Amt', 'Total_Spend', 'Avg_Spend_Ratio', 'Recency_Days']
    features = [f for f in features if f in customer_df.columns]
    
    # Log transformations for heavily skewed monetary features
    for col in ['AccountBalance', 'Avg_Transaction_Amt', 'Total_Spend']:
        if col in customer_df.columns:
            customer_df[col] = np.log1p(customer_df[col])
            
    scaler = StandardScaler()
    customer_df[features] = scaler.fit_transform(customer_df[features])
    
    # Drop intermediate columns
    if 'Last_Transaction' in customer_df.columns:
        customer_df = customer_df.drop(columns=['Last_Transaction'])
        
    return customer_df

if __name__ == "__main__":
    from src.customer_segmentation.data.indian_bank_dataset_loader import load_indian_bank_raw_data
    
    print("Loading raw data...")
    raw_df = load_indian_bank_raw_data("data/raw/bank_transactions.csv")
    print("Building features...")
    features_df = build_rfm_features(raw_df)
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/track_a_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
