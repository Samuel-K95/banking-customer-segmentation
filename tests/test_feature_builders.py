import pytest
import pandas as pd
import numpy as np
from src.features.build_indian_bank_features import build_rfm_features
from src.features.build_santander_features import build_product_portfolio_features

def test_build_rfm_features():
    # Minimal mock data to ensure the pipeline doesn't crash on standard schema
    mock_data = pd.DataFrame({
        'TransactionID': ['T1', 'T2'],
        'CustomerID': ['C1', 'C1'],
        'CustomerDOB': ['14/09/73', '14/09/73'],
        'CustGender': ['M', 'M'],
        'CustLocation': ['MUMBAI', 'MUMBAI'],
        'CustAccountBalance': [1000.0, 1500.0],
        'TransactionDate': ['1/1/16', '2/1/16'],
        'TransactionTime': [120000, 130000],
        'TransactionAmount (INR)': [100.0, 200.0]
    })
    
    result = build_rfm_features(mock_data)
    
    # Check that it aggregates to customer level
    assert len(result) == 1
    # Check that required columns exist
    assert 'Age' in result.columns
    assert 'AccountBalance' in result.columns
    assert 'Recency_Days' in result.columns

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
    
    result = build_product_portfolio_features(mock_data)
    
    # Check lengths and imputations
    assert len(result) == 2
    # Ensure missing income was imputed (median of MADRID which is 50000)
    # Note: because it's scaled, we just check no nulls exist
    assert not result['gross_income'].isnull().any()
    assert 'has_current_account' in result.columns
    assert 'is_female' in result.columns
