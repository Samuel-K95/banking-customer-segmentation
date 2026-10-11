import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from customer_segmentation.features.build_berka_features import build_berka_features
from customer_segmentation.features.build_indian_bank_features import (
    build_rfm_features,
    track_a_schema,
)
from customer_segmentation.features.build_santander_features import (
    build_product_portfolio_features,
)


def test_build_rfm_features():
    # Minimal mock data to ensure the pipeline doesn't crash on standard schema
    mock_data = pd.DataFrame({
        'TransactionID': ['T1', 'T2', 'T3'],
        'CustomerID': ['C1', 'C1', 'C2'],
        'CustomerDOB': ['14/09/73', '14/09/73', '02/03/90'],
        'CustGender': ['M', 'M', 'F'],
        'CustLocation': ['MUMBAI', 'MUMBAI', 'NEW DELHI'],
        'CustAccountBalance': [1000.0, 1500.0, 800.0],
        'TransactionDate': ['1/1/16', '2/1/16', '3/1/16'],
        'TransactionTime': [120000, 130000, 220000],
        'TransactionAmount (INR)': [100.0, 200.0, 50.0]
    })

    interpretable, model = build_rfm_features(mock_data)

    # Aggregates to one row per customer and keeps the key for joining segments back
    assert len(interpretable) == 2
    assert len(model) == 2
    assert set(interpretable['CustomerID']) == {'C1', 'C2'}
    assert list(model['CustomerID']) == list(interpretable['CustomerID'])
    for col in ['Age', 'AccountBalance', 'Recency_Days']:
        assert col in interpretable.columns

    c1 = interpretable.set_index('CustomerID').loc['C1']
    assert c1['Transaction_Count'] == 2
    assert c1['Total_Spend'] == 300.0
    assert c1['Age'] == 43  # 1973 via the century pivot, not 2073

    # NEW DELHI counts as a Tier-1 city
    assert interpretable.set_index('CustomerID').loc['C2', 'Is_Tier_1_City'] == 1


def test_track_a_schema_rejects_duplicate_customers():
    dup = pd.DataFrame({
        'CustomerID': ['C1', 'C1'],
        'Age': [30.0, 30.0],
        'AccountBalance': [1.0, 1.0],
        'Transaction_Count': [1, 1],
        'Total_Spend': [1.0, 1.0],
    })
    with pytest.raises(pa.errors.SchemaError):
        track_a_schema.validate(dup)


def test_build_santander_features():
    # Minimal mock data
    mock_data = pd.DataFrame({
        'customer_id': [1, 2],
        'age': ['25', ' 30'],
        'seniority_months': ['12', '24'],
        'gross_income': [50000.0, np.nan],
        'province_name': ['MADRID', 'MADRID'],
        'customer_segment': ['02 - PARTICULARES', '01 - TOP'],
        'gender': ['V', 'H'],
        'has_current_account': [1.0, 0.0]
    })

    interpretable, model = build_product_portfolio_features(mock_data)

    assert len(interpretable) == 2
    assert len(model) == 2
    # Missing income is imputed with the MADRID median
    assert interpretable['gross_income'].tolist() == [50000.0, 50000.0]
    assert not model['gross_income'].isnull().any()
    assert 'has_current_account' in interpretable.columns
    assert 'is_female' in interpretable.columns
    assert 'customer_id' in model.columns


def test_santander_sentinels_are_imputed():
    mock_data = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'age': ['25', '164', '40'],
        'seniority_months': ['12', '-999999', '36'],
        'gross_income': [50000.0, 60000.0, 70000.0],
        'has_current_account': [1.0, 0.0, 1.0],
    })

    interpretable, _ = build_product_portfolio_features(mock_data)

    row = interpretable.set_index('customer_id').loc[2]
    assert row['age'] == 32.5  # median of 25 and 40
    assert row['seniority_months'] == 24.0  # median of 12 and 36


def test_build_berka_features():
    trans = pd.DataFrame({
        'trans_id': [1, 2, 3, 4],
        'account_id': [10, 10, 10, 20],
        'date': [930101, 930105, 930110, 930110],
        'type': ['PRIJEM', 'VYDAJ', 'PRIJEM', 'PRIJEM'],
        'amount': [1000.0, 300.0, 500.0, 200.0],
        'balance': [1000.0, 700.0, 1200.0, 200.0],
    })
    account = pd.DataFrame({
        'account_id': [10, 20],
        'date': [921201, 930101],
    })

    interpretable, model = build_berka_features(trans, account)

    acc = interpretable.set_index('account_id')
    assert len(acc) == 2
    assert acc.loc[10, 'Transaction_Count'] == 3
    assert acc.loc[10, 'Total_Credit'] == 1500.0
    assert acc.loc[10, 'Total_Withdrawal'] == 300.0
    assert acc.loc[10, 'Account_Balance'] == 1200.0
    assert acc.loc[20, 'Recency_Days'] == 0
    # The ID is a join key, not a feature, so it must not be scaled
    assert model['account_id'].tolist() == interpretable['account_id'].tolist()
