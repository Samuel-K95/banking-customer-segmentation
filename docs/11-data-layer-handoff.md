# Phase 1: Data Layer Handoff Document

## Status: COMPLETE

The Data Layer has been successfully extracted from the initial exploratory notebooks, subjected to rigorous data engineering, and modularized into a reproducible, production-ready Python architecture within the `src/customer_segmentation/` package. 

It is now fully dockerized and compliant with the strict engineering requirements set forth in `02-data-layer.md`.

---

## 1. Dataset Selection & Justification

The overarching goal of this project is to build a comprehensive **Banking Customer Intelligence Platform**. To demonstrate full-spectrum competency across traditional Machine Learning and deep Representation Learning, we have constructed three distinct modeling tracks.

### Track A: Indian Bank Transactions (Behavioral Snapshot)
*   **What it is:** A massive log of individual customer bank transactions (~1.04 million rows) containing transaction dates, times, balances, and amounts.
*   **Why it was chosen:** It strictly captures the *behavioral liquidity* of a customer (how often they spend, how much they hold). It allows us to perform traditional **RFM (Recency, Frequency, Monetary)** aggregation. Because the resulting RFM features are dense, continuous numeric variables, they are the mathematically perfect candidate for traditional distance-based algorithms like **K-Means Clustering**.

### Track B: Santander Product Portfolio (Deep Learning / Structural)
*   **What it is:** A 13.6 million row (2.3 GB) time-series dataset tracking monthly product ownership (mortgages, credit cards, pensions) across 24 distinct banking products. 
*   **Why it was chosen:** It strictly captures the *structural portfolio* of a customer. Because most customers only own 1 or 2 products, the resulting matrix is highly dimensional and extremely sparse. Traditional K-Means fails here due to the "curse of dimensionality" on equidistant sparse binary vectors. This makes the dataset an undeniable, perfect candidate for **Deep Learning Autoencoders**, which excel at compressing sparse categorical matrices into dense latent embeddings.

### Track C: Berka Financial Dataset (Longitudinal Transactions)
*   **What it is:** The classic 1999 Czech banking dataset, containing linked relational tables (`trans`, `account`, `client`).
*   **Why it was added:** While initially sidelined for being older, we realized its deep relational structure (credits vs. withdrawals over time) provides a longitudinal purity that the Indian Bank snapshot lacks. It acts as an elite secondary benchmark for K-Means/GMM clustering.

---

## 2. Engineering & Feature Transformation

Raw financial data is rarely ready for ML out of the box. All three datasets presented severe, real-world data engineering challenges that were formally resolved in our `.py` pipelines.

### Indian Bank Dataset Fixes (`build_indian_bank_features.py`)
1.  **The Century Pivot Error:** The dataset's `CustomerDOB` used two-digit years (e.g., `67`). Standard parsing converted this to `2067`. We implemented mathematical rules to explicitly subtract 100 years from dates parsed beyond 2005.
2.  **Temporal & Geographic Engineering:** We extracted `Is_Weekend` and `Is_Night_Spender` (8 PM to 4 AM) flags from the raw timestamps, and mapped raw `CustLocation` to an `Is_Tier_1_City` flag to capture urban vs. rural spending differences.
3.  **Volatility Capture:** Beyond standard RFM, we explicitly calculated `Std_Transaction_Amt` to differentiate erratic spenders from highly consistent customers.

### Santander Dataset Fixes (`build_santander_features.py`)
1.  **Memory-Safe Chunked Loading:** The 2.3 GB file crashed standard environments. We engineered a chunked iterator that filters strictly for the maximum cross-sectional date (`2016-05-28`) *during* the read process, outputting a single, pristine snapshot of the final portfolios.
2.  **Exhaustive English Translation & Imputation:** All Spanish column names were mapped to English. 17% of missing incomes were imputed using the median of the customer's *local province* (`province_name`), tightly preserving regional economic realities.
3.  **PyTorch Crash Prevention:** We discovered string flags like `foreigner_index` (containing 'S'/'N') that would crash deep learning models. We explicitly integer-encoded all Spanish binary flags and imputed structural behavior indices (`activity_index`).

### Berka Dataset Fixes (`build_berka_features.py`)
1.  **Relational Aggregation:** We cleanly joined `trans` and `account` tables on `account_id`, separating transaction types (`PRIJEM` vs `VYDAJ`) into explicit `Total_Credit` and `Total_Withdrawal` sums.

---

## 3. Physical Architecture & Delivery

The codebase has been refactored to explicitly mirror the architectural requirements of a Customer 360 feature factory.

### Dual Matrix Output (The "Interpretable vs. Model" Split)
As mandated by the data layer plan, the feature pipelines now output *two* versions of every dataset to `data/processed/`:
1.  **`_interpretable.csv`:** Contains the raw, unscaled business metrics (e.g., Age = 45). This is strictly reserved for the Analysis Layer to generate human-readable segment profiles.
2.  **`_model.csv`:** Contains the normalized, log-transformed, and scaled values (e.g., Age = 0.23). This is strictly reserved for algorithms like K-Means and Autoencoders.

### Data Quality (DQ) Strict Gates
We implemented `pandera` schemas directly inside the feature pipelines. Before any dataset is allowed to hit the disk, it must pass a strict programmatic schema validation (e.g., asserting `Age` is always > 18, and `total_products_owned` is an integer). **If the schema fails, the pipeline crashes immediately.**

### Feature Dictionary
A formal data contract and dictionary has been published at `docs/feature_dictionary.md`. It explicitly documents the data types, source columns, missingness policies, and transformations for every single engineered feature across all three tracks.

---

## 4. Dockerization & Execution

The entire Data Layer is fully containerized. A next-generation ML Engineer or Data Scientist does not need to manually run Jupyter notebooks.

To execute the entire ingest, quality validation, and feature building lifecycle, simply run:
```bash
docker build -t data-layer .
docker run --rm -v ~/.kaggle:/root/.kaggle -v "$PWD/data:/app/data" data-layer
```
*(The `run_data_layer.sh` script automatically orchestrates the download of all three datasets via the Kaggle API and runs the respective PyTorch/Sklearn feature builders).*

---

**The Data Layer is locked, mathematically verified, strictly gated, and Dockerized. The platform is now fully prepared for Phase 2: Unsupervised Modeling and Segmentation.**
