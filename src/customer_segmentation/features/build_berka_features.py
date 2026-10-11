import os

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.preprocessing import StandardScaler

# Pandera Schema
track_c_schema = pa.DataFrameSchema({
    "Account_Balance": pa.Column(float, nullable=False, coerce=True),
    "Transaction_Count": pa.Column(int, pa.Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
    "Total_Credit": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    "Total_Withdrawal": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True)
})

def build_berka_features(trans_df: pd.DataFrame, account_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds features from the Berka transactions and accounts tables."""
    
    # 1. Clean transactions
    trans_df = trans_df.copy()
    
    # Berka dates are typically YYMMDD (e.g., 930101)
    trans_df['date'] = pd.to_datetime(trans_df['date'], format='%y%m%d', errors='coerce')
    
    # Types: "PRIJEM" (Credit), "VYDAJ" (Withdrawal), "VYBER" (Withdrawal in cash)
    trans_df['Is_Credit'] = trans_df['type'].isin(['PRIJEM', 'CREDIT']).astype(int)
    trans_df['Is_Withdrawal'] = trans_df['type'].isin(['VYDAJ', 'VYBER', 'WITHDRAWAL']).astype(int)
    
    trans_df['amount_credit'] = trans_df['amount'] * trans_df['Is_Credit']
    trans_df['amount_withdrawal'] = trans_df['amount'] * trans_df['Is_Withdrawal']
    
    max_date = trans_df['date'].max()
    
    # 2. Aggregation per account_id
    agg_dict = {
        'date': 'max',
        'trans_id': 'count',
        'balance': 'last',
        'amount_credit': 'sum',
        'amount_withdrawal': 'sum'
    }
    
    account_features = trans_df.groupby('account_id').agg(agg_dict).reset_index()
    account_features = account_features.rename(columns={
        'date': 'Last_Transaction',
        'trans_id': 'Transaction_Count',
        'balance': 'Account_Balance',
        'amount_credit': 'Total_Credit',
        'amount_withdrawal': 'Total_Withdrawal'
    })
    
    account_features['Recency_Days'] = (max_date - account_features['Last_Transaction']).dt.days
    
    # Join with account creation info
    account_df['date'] = pd.to_datetime(account_df['date'], format='%y%m%d', errors='coerce')
    account_df['Account_Age_Days'] = (max_date - account_df['date']).dt.days
    
    # Merge
    merged_df = account_features.merge(account_df[['account_id', 'Account_Age_Days']], on='account_id', how='inner')
    merged_df = merged_df.drop(columns=['Last_Transaction'])
    
    interpretable_df = merged_df.copy()
    
    interpretable_df = track_c_schema.validate(interpretable_df)
        
    model_df = interpretable_df.copy()
    
    # Log Transforms
    for col in ['Account_Balance', 'Total_Credit', 'Total_Withdrawal', 'Account_Age_Days']:
        # Shift slightly for log if negative balances exist, though usually positive
        model_df[col] = np.log1p(np.maximum(model_df[col], 0))
        
    features = [c for c in model_df.columns if c != 'account_id']
    scaler = StandardScaler()
    model_df[features] = scaler.fit_transform(model_df[features])
    
    return interpretable_df, model_df

if __name__ == "__main__":
    from customer_segmentation.data.berka_dataset_loader import load_berka_raw_data
    
    print("Loading raw Berka dataset...")
    trans_df, account_df, _ = load_berka_raw_data("data/raw/berka")
    print("Building Berka features...")
    interpretable_df, model_df = build_berka_features(trans_df, account_df)

    os.makedirs("data/processed", exist_ok=True)
    interpretable_df.to_csv("data/processed/track_c_berka_interpretable.csv", index=False)
    model_df.to_csv("data/processed/track_c_berka_model.csv", index=False)
    print("Saved interpretable and model matrices to data/processed/")
