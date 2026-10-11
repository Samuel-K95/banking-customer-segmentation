import argparse
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from customer_segmentation.features.preprocessing import prepare_model_matrix

DEFAULT_PREPROCESSOR = Path("artifacts/preprocessors/track_c.joblib")

track_c_schema = pa.DataFrameSchema({
    "Account_Balance": pa.Column(float, nullable=False, coerce=True),
    "Transaction_Count": pa.Column(int, pa.Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
    "Total_Credit": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    "Total_Withdrawal": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True)
})

def build_berka_features(
    trans_df: pd.DataFrame,
    account_df: pd.DataFrame,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build account-grain features; this does not map accounts to customers."""
    
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
    account_df = account_df.copy()
    account_df['date'] = pd.to_datetime(account_df['date'], format='%y%m%d', errors='coerce')
    account_df['Account_Age_Days'] = (max_date - account_df['date']).dt.days
    
    # Merge
    merged_df = account_features.merge(
        account_df[['account_id', 'Account_Age_Days']],
        on='account_id',
        how='inner',
        validate='one_to_one',
    )
    merged_df = merged_df.drop(columns=['Last_Transaction'])
    
    interpretable_df = merged_df.copy()
    
    interpretable_df = track_c_schema.validate(interpretable_df)
        
    features = [column for column in interpretable_df.columns if column != 'account_id']
    log_columns = ['Account_Balance', 'Total_Credit', 'Total_Withdrawal', 'Account_Age_Days']
    model_df = prepare_model_matrix(
        interpretable_df,
        track="track_c",
        identifier_columns=["account_id"],
        log1p_columns=log_columns,
        clip_nonnegative_columns=log_columns,
        scaled_features=features,
        artifact_path=preprocessor_path,
        fit=fit_preprocessor,
    )

    return interpretable_df, model_df


def main() -> None:
    from customer_segmentation.data.berka_dataset_loader import load_berka_raw_data

    parser = argparse.ArgumentParser(description="Build Track C Berka account-level features.")
    parser.add_argument("--preprocessor", default=str(DEFAULT_PREPROCESSOR))
    parser.add_argument("--reuse-preprocessor", action="store_true")
    args = parser.parse_args()

    print("Loading raw Berka dataset...")
    trans_df, account_df, _ = load_berka_raw_data("data/raw/berka")
    print("Building Berka account-level features...")
    interpretable_df, model_df = build_berka_features(
        trans_df,
        account_df,
        preprocessor_path=args.preprocessor,
        fit_preprocessor=not args.reuse_preprocessor,
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    interpretable_df.to_csv("data/processed/track_c_berka_interpretable.csv", index=False)
    model_df.to_csv("data/processed/track_c_berka_model.csv", index=False)
    print(f"Saved Track C account matrices and preprocessor to data/processed/ and {args.preprocessor}")


if __name__ == "__main__":
    main()
