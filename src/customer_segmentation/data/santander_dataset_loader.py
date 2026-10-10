import pandas as pd
import numpy as np
from pathlib import Path

def load_santander_raw_data(data_path: str = "data/raw/santander/train_ver2.csv", target_date: str = '2016-05-28') -> pd.DataFrame:
    """
    Loads the Santander dataset efficiently by:
    1. Using a strict subset of columns with defined dtypes.
    2. Loading in chunks to avoid memory issues (2.3GB file).
    3. Translating Spanish column names to English.
    4. Filtering for a single snapshot date (cross-sectional).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    dtypes = {
        'ncodpers': np.int32, 'age': object, 'antiguedad': object, 'renta': np.float32,
        'ind_nuevo': np.float32, 'indrel': np.float32, 'indrel_1mes': object, 'indext': object,
        'conyuemp': object, 'indfall': object, 'tipodom': np.float32, 'ind_actividad_cliente': np.float32,
        'segmento': object, 'sexo': object, 'tiprel_1mes': object,
        'ind_cco_fin_ult1': np.float32, 'ind_cder_fin_ult1': np.float32, 'ind_cno_fin_ult1': np.float32,
        'ind_ctju_fin_ult1': np.float32, 'ind_ctma_fin_ult1': np.float32, 'ind_ctop_fin_ult1': np.float32,
        'ind_ctpp_fin_ult1': np.float32, 'ind_deco_fin_ult1': np.float32, 'ind_deme_fin_ult1': np.float32,
        'ind_dela_fin_ult1': np.float32, 'ind_ecue_fin_ult1': np.float32, 'ind_fond_fin_ult1': np.float32,
        'ind_hip_fin_ult1': np.float32, 'ind_plan_fin_ult1': np.float32, 'ind_pres_fin_ult1': np.float32,
        'ind_reca_fin_ult1': np.float32, 'ind_tjcr_fin_ult1': np.float32, 'ind_valo_fin_ult1': np.float32,
        'ind_viv_fin_ult1': np.float32, 'ind_nomina_ult1': np.float32, 'ind_nom_pens_ult1': np.float32,
        'ind_recibo_ult1': np.float32, 'nomprov': str, 'fecha_dato': str
    }

    rename_dict = {
        'ncodpers': 'customer_id', 'age': 'age', 'antiguedad': 'seniority_months', 'renta': 'gross_income',
        'segmento': 'customer_segment', 'sexo': 'gender', 'nomprov': 'province_name', 'fecha_dato': 'snapshot_date',
        'ind_nuevo': 'new_customer_index', 'indrel': 'customer_type', 'indrel_1mes': 'customer_type_primary',
        'indext': 'foreigner_index', 'conyuemp': 'spouse_index', 'indfall': 'deceased_index', 'tipodom': 'address_type',
        'ind_actividad_cliente': 'activity_index', 'tiprel_1mes': 'customer_relation_type',
        'ind_cco_fin_ult1': 'has_current_account', 'ind_cder_fin_ult1': 'has_derivada_account',
        'ind_cno_fin_ult1': 'has_payroll_account', 'ind_ctju_fin_ult1': 'has_junior_account',
        'ind_ctma_fin_ult1': 'has_mas_particular_account', 'ind_ctop_fin_ult1': 'has_particular_account',
        'ind_ctpp_fin_ult1': 'has_particular_plus_account', 'ind_deco_fin_ult1': 'has_short_term_deposits',
        'ind_deme_fin_ult1': 'has_medium_term_deposits', 'ind_dela_fin_ult1': 'has_long_term_deposits',
        'ind_ecue_fin_ult1': 'has_e_account', 'ind_fond_fin_ult1': 'has_funds', 'ind_hip_fin_ult1': 'has_mortgage',
        'ind_plan_fin_ult1': 'has_pensions', 'ind_pres_fin_ult1': 'has_loans', 'ind_reca_fin_ult1': 'has_taxes',
        'ind_tjcr_fin_ult1': 'has_credit_card', 'ind_valo_fin_ult1': 'has_securities', 'ind_viv_fin_ult1': 'has_home_account',
        'ind_nomina_ult1': 'has_payroll', 'ind_nom_pens_ult1': 'has_pensions_nom', 'ind_recibo_ult1': 'has_direct_debit'
    }

    chunksize = 1000000
    chunks = []
    
    for chunk in pd.read_csv(path, dtype=dtypes, usecols=list(dtypes.keys()), chunksize=chunksize):
        chunk.rename(columns=rename_dict, inplace=True)
        if target_date:
            filtered_chunk = chunk[chunk['snapshot_date'] == target_date]
            chunks.append(filtered_chunk)
        else:
            chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)
