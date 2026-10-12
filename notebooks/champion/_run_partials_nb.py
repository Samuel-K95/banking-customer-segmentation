from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables, load_berka_raw_data, resolve_berka_dir, _read_table,
)
from customer_segmentation.features.build_berka_segmentation_features import (
    MODEL_COLUMNS, build_berka_segmentation_features_asof, transform_with_train_scaler,
)
from customer_segmentation.models.champion import ChampionModel
from customer_segmentation.models.metrics import evaluate_clustering, size_sanity
from customer_segmentation.models.stability import multi_seed_ari, subsample_ari

CHAMP_DIR = ROOT / 'artifacts' / 'segmentation' / 'champion'
OUT_DIR = ROOT / 'artifacts' / 'segmentation' / 'research' / 'partials_fix_scratch'
OUT_DIR.mkdir(parents=True, exist_ok=True)

champ = ChampionModel.load(CHAMP_DIR)
metrics = champ.metrics or {}
as_of = str(metrics.get('as_of_date') or '1997-12-31')
min_tenure = int(metrics.get('minimum_tenure_days') or 365)
n_clusters = int(champ.n_clusters)
split = pd.read_csv(CHAMP_DIR / 'account_split.csv')
assign = pd.read_csv(CHAMP_DIR / 'assignments.csv')

path = resolve_berka_dir()
trans, account, client = load_berka_raw_data(path)
loan, card, disp = load_berka_product_tables(path)
interpretable = build_berka_segmentation_features_asof(
    trans, account, loan, card, disp, as_of_date=as_of, minimum_tenure_days=min_tenure,
)
df = interpretable.merge(split, on='account_id', how='inner').merge(
    assign[['account_id', 'segment_id', 'confidence']], on='account_id', how='inner'
)
train_df = df[df['split'] == 'train'].copy()
hold_df = df[df['split'] == 'holdout'].copy()
X_train = transform_with_train_scaler(train_df, champ.scaler)
X_hold = transform_with_train_scaler(hold_df, champ.scaler)
z_train = champ.encode(X_train)
z_hold = champ.encode(X_hold)
print('loaded', champ.run_id, len(df), len(train_df), len(hold_df))

def fit_km(X, seed, k=n_clusters):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)

champ_hold_labels = champ.kmeans.predict(z_hold)
champ_hold_metrics = evaluate_clustering(z_hold, champ_hold_labels)
champ_train_metrics = evaluate_clustering(z_train, champ.kmeans.predict(z_train))
seed_stability = multi_seed_ari(lambda X, s: fit_km(X, s), z_train, n_seeds=5, base_seed=42)
sub_stability = subsample_ari(lambda X, s: fit_km(X, s), z_train, frac=0.75, n_rounds=5, random_state=42)

km6 = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(X_train)
km6_hold = evaluate_clustering(X_hold, km6.predict(X_hold))
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=3)
gmm.fit(X_train)
gmm_hold = evaluate_clustering(X_hold, gmm.predict(X_hold))
hier_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_train)
hier_train = evaluate_clustering(X_train, hier_labels)
km2 = KMeans(n_clusters=2, n_init=20, random_state=42).fit(X_train)
km2_hold = evaluate_clustering(X_hold, km2.predict(X_hold))

board = pd.DataFrame([
    {'run_id': champ.run_id, 'family': 'ae_kmeans', 'k': n_clusters, 'latent_dim': 4, 'split': 'holdout',
     'silhouette': round(champ_hold_metrics['silhouette'], 4), 'db_index': round(champ_hold_metrics['db_index'], 4),
     'ch_index': round(champ_hold_metrics['ch_index'], 0), 'seed_ari_mean': round(seed_stability['mean_ari'], 4),
     'subsample_ari_mean': round(sub_stability['mean_ari'], 4), 'role': 'product_champion_leakage_safe'},
    {'run_id': 'c_kmeans_k6_scaled_leakage_safe', 'family': 'kmeans', 'k': 6, 'latent_dim': '-', 'split': 'holdout',
     'silhouette': round(km6_hold['silhouette'], 4), 'db_index': round(km6_hold['db_index'], 4),
     'ch_index': round(km6_hold['ch_index'], 0), 'seed_ari_mean': None, 'subsample_ari_mean': None,
     'role': 'traditional_baseline_same_matrix'},
    {'run_id': 'c_gmm_k6_scaled_leakage_safe', 'family': 'gmm', 'k': 6, 'latent_dim': '-', 'split': 'holdout',
     'silhouette': round(gmm_hold['silhouette'], 4), 'db_index': round(gmm_hold['db_index'], 4),
     'ch_index': round(gmm_hold['ch_index'], 0), 'seed_ari_mean': None, 'subsample_ari_mean': None,
     'role': 'traditional_baseline_same_matrix'},
    {'run_id': 'c_hier_k6_scaled_leakage_safe', 'family': 'hierarchical', 'k': 6, 'latent_dim': '-', 'split': 'train',
     'silhouette': round(hier_train['silhouette'], 4), 'db_index': round(hier_train['db_index'], 4),
     'ch_index': round(hier_train['ch_index'], 0), 'seed_ari_mean': None, 'subsample_ari_mean': None,
     'role': 'traditional_baseline_train_only'},
    {'run_id': 'c_kmeans_k2_leakage_safe', 'family': 'kmeans', 'k': 2, 'latent_dim': '-', 'split': 'holdout',
     'silhouette': round(km2_hold['silhouette'], 4), 'db_index': round(km2_hold['db_index'], 4),
     'ch_index': round(km2_hold['ch_index'], 0), 'seed_ari_mean': None, 'subsample_ari_mean': None,
     'role': 'research_twin_same_matrix'},
])
print(board.to_string(index=False))
print('seed', seed_stability)
print('subsample', sub_stability)

def berka_age_gender(birth_number, as_of_year=1997):
    bn = int(birth_number)
    yy, mm, dd = bn // 10000, (bn // 100) % 100, bn % 100
    is_female = 0
    if mm > 50:
        mm -= 50
        is_female = 1
    year = 1900 + yy
    age = max(as_of_year - year, 0)
    return age, is_female

disp_t = disp.copy() if disp is not None else _read_table(path, 'disp')
owners = disp_t[disp_t['type'].astype(str).str.upper().eq('OWNER')].copy()
owners = owners.merge(client[['client_id', 'birth_number']], on='client_id', how='left')
ages, females = zip(*[berka_age_gender(b) for b in owners['birth_number'].fillna(0).astype(int)])
owners['age'] = ages
owners['is_female'] = females
owners = owners.merge(account[['account_id', 'district_id']], on='account_id', how='left')
demo = owners.groupby('account_id', as_index=False).agg(
    age=('age', 'first'), is_female=('is_female', 'first'), district_id=('district_id', 'first')
)
demo_df = train_df.merge(demo, on='account_id', how='left')
demo_df['age'] = demo_df['age'].fillna(demo_df['age'].median())
demo_df['is_female'] = demo_df['is_female'].fillna(0).astype(int)
demo_df['district_id'] = demo_df['district_id'].fillna(-1).astype(int)
top_d = demo_df['district_id'].value_counts().head(15).index
demo_df['district_top'] = demo_df['district_id'].where(demo_df['district_id'].isin(top_d), other=-1)
dist_oh = pd.get_dummies(demo_df['district_top'], prefix='dist')
beh_cols = list(MODEL_COLUMNS)
X_beh = StandardScaler().fit_transform(demo_df[beh_cols].astype(float))
X_full = StandardScaler().fit_transform(np.hstack([
    demo_df[beh_cols].astype(float).values,
    demo_df[['age','is_female']].astype(float).values,
    dist_oh.astype(float).values,
]))
lab_beh = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(X_beh)
lab_full = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(X_full)
ablation = {
    'n_train': int(len(demo_df)), 'k': n_clusters,
    'behavior_features': beh_cols,
    'added_demographics': ['age', 'is_female', 'district_top15_onehot'],
    'ari_behavior_vs_behavior_plus_demo': float(adjusted_rand_score(lab_beh, lab_full)),
    'ari_champion_vs_behavior_kmeans': float(adjusted_rand_score(demo_df['segment_id'].values, lab_beh)),
    'ari_champion_vs_behavior_plus_demo_kmeans': float(adjusted_rand_score(demo_df['segment_id'].values, lab_full)),
    'interpretation': 'High ARI(behavior, behavior+demo) means demographics do not rewrite segments; champion remains behavior-only.',
}
print('ablation', json.dumps(ablation, indent=2))

name_map = {int(k): v['name'] for k, v in (champ.catalog or {}).get('segments', {}).items()}
rows = []
for sid, g in df.groupby('segment_id'):
    rows.append({
        'segment_id': int(sid), 'segment_name': name_map.get(int(sid)), 'n': int(len(g)),
        'pct': round(100 * len(g) / len(df), 2),
        'product_penetration': {
            'has_loan_rate': round(float(g['has_loan'].mean()), 4),
            'has_card_rate': round(float(g['has_card'].mean()), 4),
            'both_products_rate': round(float(((g['has_loan']==1)&(g['has_card']==1)).mean()), 4),
            'neither_product_rate': round(float(((g['has_loan']==0)&(g['has_card']==0)).mean()), 4),
        },
        'lifecycle_mix': {
            'median_account_age_days': round(float(g['Account_Age_Days'].median()), 1),
            'median_recency_days': round(float(g['Recency_Days'].median()), 1),
            'share_recency_gt_30': round(float((g['Recency_Days']>30).mean()), 4),
            'share_tenure_lt_730': round(float((g['Account_Age_Days']<730).mean()), 4),
            'share_negative_balance': round(float((g['Account_Balance']<0).mean()), 4),
        },
        'fairness_notes': {
            'usage': 'CRM/journey only — not credit eligibility or pricing',
            'size_policy': 'exception_queue_not_mass_campaign' if len(g)/len(df) < 0.03 else 'mass_campaign_eligible_with_guardrails',
        },
    })

payload = {
    'champion_run_id': champ.run_id, 'model_version': champ.model_version, 'as_of_date': as_of,
    'holdout_metrics_shared': champ_hold_metrics, 'train_metrics_shared': champ_train_metrics,
    'stability_seed_ari': seed_stability, 'stability_subsample_ari': sub_stability,
    'comparison_board': board.to_dict(orient='records'), 'demographic_ablation': ablation,
    'profile_enrichment_preview': {'packs_preview': rows},
    'k_justification_numbers': {
        'holdout_silhouette': champ_hold_metrics['silhouette'],
        'holdout_db_index': champ_hold_metrics['db_index'],
        'holdout_ch_index': champ_hold_metrics['ch_index'],
        'seed_ari_mean': seed_stability['mean_ari'],
        'subsample_ari_mean': sub_stability['mean_ari'],
        'raw_kmeans_k6_holdout_silhouette': km6_hold['silhouette'],
        'gmm_k6_holdout_silhouette': gmm_hold['silhouette'],
        'hier_k6_train_silhouette': hier_train['silhouette'],
        'kmeans_k2_holdout_silhouette': km2_hold['silhouette'],
        'n_eligible': int(len(df)), 'n_train': int(len(train_df)), 'n_holdout': int(len(hold_df)),
        'size': size_sanity(df['segment_id'].values),
    },
}
(OUT_DIR / 'partials_fix_notebook_results.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
board.to_csv(OUT_DIR / 'comparison_board_leakage_safe_enriched.csv', index=False)
print('wrote', OUT_DIR)
