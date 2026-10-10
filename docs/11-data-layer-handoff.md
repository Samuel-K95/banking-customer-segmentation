# Phase 1: Data Layer Handoff Document

## Status: COMPLETE

The Data Layer has been successfully extracted from the initial exploratory notebooks, subjected to rigorous data engineering, and modularized into a reproducible, production-ready Python architecture within the `src/customer_segmentation/` package. 

---

## 1. Dataset Selection & Justification

The overarching goal of this project is to build a comprehensive **Banking Customer Intelligence Platform**. To demonstrate full-spectrum competency across traditional Machine Learning and deep Representation Learning, we deliberately selected two distinct, complementary financial datasets.

### Track A: Indian Bank Transactions (Classical ML / Behavioral)
*   **What it is:** A massive log of individual customer bank transactions (~1.04 million rows) containing transaction dates, times, balances, and amounts.
*   **Why it was chosen:** It strictly captures the *behavioral liquidity* of a customer (how often they spend, how much they hold). It allows us to perform traditional **RFM (Recency, Frequency, Monetary)** aggregation. Because the resulting RFM features are dense, continuous numeric variables, they are the mathematically perfect candidate for traditional distance-based algorithms like **K-Means Clustering**.

### Track B: Santander Product Portfolio (Deep Learning / Structural)
*   **What it is:** A 13.6 million row (2.3 GB) time-series dataset tracking monthly product ownership (mortgages, credit cards, pensions) across 24 distinct banking products. 
*   **Why it was chosen:** It strictly captures the *structural portfolio* of a customer. Because most customers only own 1 or 2 products, the resulting matrix is highly dimensional and extremely sparse. Traditional K-Means fails here due to the "curse of dimensionality" on equidistant sparse binary vectors. This makes the dataset an undeniable, perfect candidate for **Deep Learning Autoencoders**, which excel at compressing sparse categorical matrices into dense latent embeddings.

*(Note: Other datasets, like the 1999 Berka dataset and IBM AML synthetic data, were rejected for being either too antiquated for modern digital banking or heavily biased toward fraud detection rather than behavioral segmentation).*

---

## 2. Engineering & Feature Transformation

Raw financial data is rarely ready for ML out of the box. Both datasets presented severe, real-world data engineering challenges that were formally resolved in our `.py` pipelines.

### Indian Bank Dataset Fixes (`build_indian_bank_features.py`)
1.  **The Century Pivot Error:** The dataset's `CustomerDOB` used two-digit years (e.g., `67`). Standard parsing converted this to `2067`, resulting in impossible negative ages. We implemented mathematical rules to explicitly subtract 100 years from dates parsed beyond 2005.
2.  **Temporal Context (2016 Base Year):** Customers' ages were calculated relative to `2016` (when the transactions actually occurred) rather than the present day, preventing a 10-year artificial skew in customer age segments.
3.  **RFM Aggregation:** 1 million raw transactions were successfully collapsed into unique, customer-level profiles measuring `Recency_Days`, `Transaction_Count`, `Total_Spend`, and a custom financial health proxy (`Spend_to_Balance_Ratio`).
4.  **Skewness Normalization:** Extreme wealth outliers (balances of 115M+ INR) were handled via `np.log1p` transforms, followed by strict `StandardScaler` application to ensure Euclidean compatibility for K-Means.

### Santander Dataset Fixes (`build_santander_features.py`)
1.  **Memory-Safe Chunked Loading:** The 2.3 GB file crashed standard environments. We engineered a chunked iterator that filters strictly for the maximum cross-sectional date (`2016-05-28`) *during* the read process, dropping the time-series dimension and outputting a single, pristine snapshot of the final portfolios.
2.  **Exhaustive English Translation:** All Spanish column names and product variables (e.g., `ind_cco_fin_ult1`, `renta`) were mapped via a comprehensive translation dictionary to clean English (`has_current_account`, `gross_income`).
3.  **Regional Income Imputation:** 17% of customers were missing income data. Rather than dropping them (which introduces bias) or using a lazy global median, we imputed missing incomes by calculating the specific median income for the customer's *local province* (`province_name`), tightly preserving regional economic realities.
4.  **Dual-Track Scaling:** We strictly separated the data into two tracks prior to ML ingestion: Demographics (Age, Income, Seniority) were passed through `StandardScaler`, while the 24 Product vectors and one-hot segments were explicitly preserved as strict `0` or `1` binary inputs to maintain stability for the PyTorch Neural Network.

---

## 3. Physical Architecture & Repository Layout

The codebase has been refactored to explicitly mirror the architectural requirements, isolating the raw source data from the downstream ML matrices.

### Data Storage
*   `data/raw/`: Exclusively holds the untouched source files (`bank_transactions.csv`, `santander/train_ver2.csv`).
*   `data/processed/`: The designated sink for the final `.csv` feature matrices.

### Python Modules
The notebook exploratory code was formalized into the `src/customer_segmentation/` package:
*   `data/indian_bank_dataset_loader.py` & `data/santander_dataset_loader.py`: Handles I/O, pathing, chunking, and renaming.
*   `features/build_indian_bank_features.py` & `features/build_santander_features.py`: Imports the loaders and applies the exact normalization math.

---

## 4. Quality Assurance & Testing

Sanity checks and CI/CD ready unit tests were implemented in `tests/test_data_loaders.py` and `tests/test_feature_builders.py`. 

Crucially, the feature builder tests inject *isolated, mocked DataFrames* into the pipeline. This ensures that the complex algorithmic transformations (century pivots, median imputations) are continuously verified in milliseconds without the need to load the multi-gigabyte raw datasets into memory during a test suite run.

---

## 5. Generated Deliverables

Execution of the data layer via `uv run python -m src.customer_segmentation.features.build_...` successfully generates two immutable artifacts:

1.  `data/processed/track_a_features.csv`: The dense RFM matrix for K-Means.
2.  `data/processed/track_b_features.csv`: The sparse portfolio matrix for PyTorch.

**The Data Layer is locked and mathematically verified. The platform is now fully prepared for Phase 2: Unsupervised Modeling and Segmentation.**
