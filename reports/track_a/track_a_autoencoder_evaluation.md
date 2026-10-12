# Track A Autoencoder Representation Evaluation

**Run date:** 2026-10-11
**Population:** 837,951 validated customers
**Methods:** CPU PyTorch autoencoders with latent K-Means, a spending-ratio feature ablation, and traditional eight-feature K-Means

## Executive summary

The evaluation used the same two 50,000-customer training manifests and shared 3,000-customer holdout for the nine-feature and eight-feature-without-ratio comparisons. Four AE fits completed, producing 12 AE–K-Means candidates; six additional traditional eight-feature K-Means candidates completed. All 42 metric-space records were valid.

The AE results do not establish consistent added value. AE-9 k=2 scored below traditional nine-feature K-Means/GMM in the common original feature space; k=3 results varied by seed. Cross-seed AE assignment ARI was 0.3286 for AE-9 k=2 and 0.2022 for AE-8 k=2, despite stronger latent-space silhouette values. Latent-space scores are not directly comparable to original-space scores. Several AE partitions strongly separate single-transaction customers, and the eight-feature ablation changes assignments. No champion is selected.

## Architecture and optimization

The configured models were 9→32→16→4→16→32→9 and 8→32→16→4→16→32→8. Hidden layers use ReLU; the bottleneck and output are linear. Training used AdamW (`learning_rate=0.001`, `weight_decay=0.0001`), SmoothL1/Huber loss (`beta=1`), batch size 512, at most 80 epochs, patience 10, and `min_delta=1e-5`. Seeds were 42 and 2024. Each 50,000-customer training manifest was split deterministically into 45,000 internal-training and 5,000 internal-validation IDs; the split seed is the model seed plus 91,731. The best internal-validation checkpoint was restored.

Huber loss limits the influence of extreme residuals but does not eliminate the ratio tail. The largest initial globally standardized `Avg_Spend_Ratio` value was 358.78 standard deviations. Reconstruction losses and errors are diagnostics, not clustering-quality scores.

## Data and feature controls

| Split | Manifest | Rows | Ordered-ID SHA-256 |
|---|---|---:|---|
| Train, seed 42 | `artifacts/experiments/samples/track_a_comparison_train_seed42_n50000_d23d078cbe1c.csv` | 50,000 | `d23d078cbe1c5db176d730651032b6130e81891340bcc1ca22592ae9b6ebe102` |
| Train, seed 2024 | `artifacts/experiments/samples/track_a_comparison_train_seed2024_n50000_889a5996e041.csv` | 50,000 | `889a5996e0416f49a96f0baca9c9781e8d3f47b1ed0092bd86f23817ce1e9ae8` |
| Shared held-out evaluation | `artifacts/experiments/samples/track_a_evaluation_n3000_5496dbb5ee72.csv` | 3,000 | `5496dbb5ee72aed5b66bea49410f00624c724173413606edf4ce23085c5a16d3` |

Each training manifest is disjoint from evaluation. Internal splits use training IDs only. `CustomerID` is an alignment key and never an input; `Age`, `Gender_M`, and `Is_Tier_1_City` are excluded from clustering. Models consumed the existing globally transformed feature matrix as-is, converted to float32 for PyTorch batches; no second feature scaler was applied. Each latent StandardScaler was fit on that seed's 50,000 training embeddings only.

The historical global scaler was fit on all 837,951 validated customers before the baseline split. Comparisons using that matrix therefore inherit feature-distribution leakage and are transductive with respect to the scaler, even though the AE internal validation and latent scaler use only training IDs. This experiment does not claim an independent inductive holdout.

## Execution and environment

The synthetic smoke passed forward/backward, finite-loss, checkpoint reload, deterministic encoding, train-only latent scaling, and K-Means fit/predict. A real-data nine-feature k=2 smoke completed. The planned comparison completed four AE fits, 12 AE–K-Means candidates, and six traditional eight-feature K-Means candidates (42 metric-space records total). Execution used CPU-only PyTorch 2.13.0+cpu, one Torch thread, Python 3.12.10, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0, and threadpoolctl 3.7.0. CSV tracking was used because MLflow was unavailable.

## Training and reconstruction results

| Feature set | Seed | Best epoch | Epochs run | Best train Huber | Best validation Huber | Last train Huber | Last validation Huber | Stopped early |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Nine-feature | 42 | 75 | 80 | 0.00462438 | 0.01027118 | 0.00457165 | 0.01041152 | No |
| Nine-feature | 2024 | 80 | 80 | 0.00478101 | 0.00305613 | 0.00478101 | 0.00305613 | No |
| Eight-feature, no ratio | 42 | 79 | 80 | 0.00656385 | 0.00708462 | 0.00656680 | 0.00711089 | No |
| Eight-feature, no ratio | 2024 | 78 | 80 | 0.00537475 | 0.00550406 | 0.00533723 | 0.00552973 | No |

Internal-validation feature reconstruction MAE / RMSE in the initial globally standardized feature units:

| Feature | AE-9 seed 42 | AE-9 seed 2024 | AE-8 seed 42 | AE-8 seed 2024 |
|---|---:|---:|---:|---:|
| `AccountBalance` | 0.03653 / 0.10480 | 0.02626 / 0.05608 | 0.02163 / 0.03374 | 0.08259 / 0.24500 |
| `Transaction_Count` | 0.01574 / 0.03282 | 0.01087 / 0.02995 | 0.02339 / 0.02791 | 0.00886 / 0.02367 |
| `Avg_Transaction_Amt` | 0.02175 / 0.05186 | 0.02425 / 0.05480 | 0.02284 / 0.05146 | 0.02202 / 0.05187 |
| `Std_Transaction_Amt` | 0.04012 / 0.13302 | 0.03576 / 0.11873 | 0.03958 / 0.11971 | 0.03763 / 0.12489 |
| `Total_Spend` | 0.02095 / 0.05024 | 0.02429 / 0.05301 | 0.02036 / 0.04919 | 0.02199 / 0.05109 |
| `Avg_Spend_Ratio` | **0.08895 / 3.07103** | 0.01796 / 0.10196 | — | — |
| `Recency_Days` | 0.02255 / 0.04522 | 0.01761 / 0.03425 | 0.01304 / 0.02607 | 0.01539 / 0.03127 |
| `Weekend_Spend_Ratio` | 0.01101 / 0.08299 | 0.02391 / 0.10001 | 0.02959 / 0.10460 | 0.02420 / 0.09915 |
| `Night_Spend_Ratio` | 0.03568 / 0.13947 | 0.03917 / 0.12165 | **0.11139 / 0.29657** | 0.01361 / 0.02950 |

The AE-9 seed-42 ratio RMSE and AE-8 seed-42 night-ratio RMSE illustrate seed/feature sensitivity. Complete unrounded train and validation errors are in `artifacts/experiments/run_metadata/`; per-epoch traces are in `artifacts/experiments/autoencoder_history/`.

## Clustering results and comparable baselines

All metrics use the shared 3,000-customer evaluation set. Silhouette uses all 3,000 rows; DB and CH use the same rows. Memberships are counts (percentages; fraction of cluster with one transaction).

| Candidate | Seed | k | Original 9-feature S / DB / CH | Original 8-feature S / DB / CH | Scaled latent 4D S / DB / CH | Evaluation memberships | Smallest % | Fit seconds | Peak WS MiB | Cross-seed ARI |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|
| AE-9 + K-Means | 42 | 2 | 0.1701 / 2.6293 / 370.7 | — | 0.4101 / 1.1770 / 1,724.8 | 1,836 (61.2%; 1txn 95.8%); 1,164 (38.8%; 1txn 64.6%) | 38.80 | 84.45 | 425.5 | 0.3286 |
| AE-9 + K-Means | 42 | 3 | 0.2539 / 1.6532 / 773.0 | — | 0.5203 / 0.7506 / 2,386.1 | 707 (23.6%; 1txn 100%); 489 (16.3%; 1txn 0%); 1,804 (60.1%; 1txn 100%) | 16.30 | 84.60 | 434.1 | 0.9720 |
| AE-9 + K-Means | 42 | 4 | 0.2305 / 1.7095 / 692.6 | — | 0.4647 / 0.7658 / 2,671.8 | 1,066 (35.5%; 1txn 100%); 686 (22.9%; 1txn 100%); 489 (16.3%; 1txn 0%); 759 (25.3%; 1txn 100%) | 16.30 | 84.75 | 434.3 | 0.7999 |
| AE-9 + K-Means | 2024 | 2 | 0.2545 / 1.8514 / 605.5 | — | 0.5547 / 0.7591 / 3,346.1 | 716 (23.9%; 1txn 31.7%); 2,284 (76.1%; 1txn 100%) | 23.87 | 78.71 | 443.6 | 0.3286 |
| AE-9 + K-Means | 2024 | 3 | 0.2537 / 1.5991 / 789.7 | — | 0.6393 / 0.5550 / 6,028.7 | 489 (16.3%; 1txn 0%); 1,823 (60.8%; 1txn 100%); 688 (22.9%; 1txn 100%) | 16.30 | 78.83 | 446.2 | 0.9720 |
| AE-9 + K-Means | 2024 | 4 | 0.2655 / 1.4705 / 754.2 | — | 0.6110 / 0.5328 / 6,462.5 | 489 (16.3%; 1txn 0%); 1,237 (41.2%; 1txn 100%); 688 (22.9%; 1txn 100%); 586 (19.5%; 1txn 100%) | 16.30 | 78.95 | 447.4 | 0.7999 |
| AE-8 + K-Means | 42 | 2 | 0.3308 / 1.1771 / 920.5 | 0.3309 / 1.1673 / 958.9 | 0.4560 / 0.7840 / 1,677.5 | 489 (16.3%; 1txn 0%); 2,511 (83.7%; 1txn 100%) | 16.30 | 77.07 | 456.0 | 0.2022 |
| AE-8 + K-Means | 42 | 3 | 0.2527 / 1.5941 / 816.7 | 0.2551 / 1.5859 / 857.3 | 0.4723 / 0.8060 / 2,462.7 | 489 (16.3%; 1txn 0%); 1,698 (56.6%; 1txn 100%); 813 (27.1%; 1txn 100%) | 16.30 | 77.75 | 457.9 | 1.0000 |
| AE-8 + K-Means | 42 | 4 | 0.2085 / 1.7183 / 698.5 | 0.2096 / 1.7077 / 736.8 | 0.4555 / 0.7936 / 2,889.9 | 806 (26.9%; 1txn 100%); 949 (31.6%; 1txn 100%); 489 (16.3%; 1txn 0%); 756 (25.2%; 1txn 100%) | 16.30 | 77.67 | 459.3 | 0.5937 |
| AE-8 + K-Means | 2024 | 2 | 0.1939 / 2.2352 / 525.8 | 0.1949 / 2.2241 / 545.5 | 0.5523 / 0.9263 / 2,821.5 | 1,696 (56.5%; 1txn 100%); 1,304 (43.5%; 1txn 62.5%) | 43.47 | 76.22 | 467.6 | 0.2022 |
| AE-8 + K-Means | 2024 | 3 | 0.2527 / 1.5941 / 816.7 | 0.2551 / 1.5859 / 857.3 | 0.7054 / 0.4224 / 8,653.8 | 489 (16.3%; 1txn 0%); 1,698 (56.6%; 1txn 100%); 813 (27.1%; 1txn 100%) | 16.30 | 76.39 | 470.4 | 1.0000 |
| AE-8 + K-Means | 2024 | 4 | 0.2591 / 1.5659 / 719.7 | 0.2607 / 1.5567 / 759.8 | 0.6074 / 0.5591 / 9,073.8 | 813 (27.1%; 1txn 100%); 1,198 (39.9%; 1txn 100%); 489 (16.3%; 1txn 0%); 500 (16.7%; 1txn 100%) | 16.30 | 76.50 | 472.1 | 0.5937 |
| Traditional K-Means-8 | 42 | 2 | 0.3327 / 1.1637 / 922.8 | 0.3328 / 1.1539 / 961.3 | — | 481 (16.0%; 1txn 0%); 2,519 (84.0%; 1txn 99.7%) | 16.03 | 0.65 | 488.9 | 1.0000 |
| Traditional K-Means-8 | 42 | 3 | 0.2215 / 1.6063 / 895.8 | 0.2228 / 1.5965 / 941.9 | — | 1,177 (39.2%; 1txn 98.9%); 475 (15.8%; 1txn 0%); 1,348 (44.9%; 1txn 99.9%) | 15.83 | 0.95 | 482.3 | 0.9574 |
| Traditional K-Means-8 | 42 | 4 | 0.2391 / 1.4593 / 819.9 | 0.2403 / 1.4482 / 868.4 | — | 841 (28.0%; 1txn 98.6%); 1,076 (35.9%; 1txn 99.8%); 475 (15.8%; 1txn 0%); 608 (20.3%; 1txn 100%) | 15.83 | 1.36 | 481.3 | 0.9750 |
| Traditional K-Means-8 | 2024 | 2 | 0.3327 / 1.1637 / 922.8 | 0.3328 / 1.1539 / 961.3 | — | 481 (16.0%; 1txn 0%); 2,519 (84.0%; 1txn 99.7%) | 16.03 | 1.09 | 480.3 | 1.0000 |
| Traditional K-Means-8 | 2024 | 3 | 0.2216 / 1.6070 / 895.8 | 0.2229 / 1.5973 / 941.8 | — | 1,345 (44.8%; 1txn 99.9%); 477 (15.9%; 1txn 0%); 1,178 (39.3%; 1txn 99.1%) | 15.90 | 0.85 | 481.5 | 0.9574 |
| Traditional K-Means-8 | 2024 | 4 | 0.2392 / 1.4587 / 820.2 | 0.2403 / 1.4475 / 868.6 | — | 1,061 (35.4%; 1txn 99.8%); 614 (20.5%; 1txn 99.8%); 850 (28.3%; 1txn 98.7%); 475 (15.8%; 1txn 0%) | 15.83 | 1.12 | 483.9 | 0.9750 |

The AE rows share a fitted network by seed across k=2/3/4; candidate times include that common fit and should not be added as independent network runtimes. The original nine-feature comparisons show that AE-9 k=2 is below traditional K-Means on both seeds; k=3 is higher on silhouette but lower on DB/CH for some comparisons and has low AE-to-K-Means ARI (~0.27). Latent scores are not ranked against original-space scores.

## Matched traditional baselines and GMM alternative

The direct original nine-feature baseline is the traditional nine-feature K-Means/GMM from the same evaluation IDs. Its seed-specific values are in [`track_a_traditional_clustering_baselines.md`](track_a_traditional_clustering_baselines.md). Ratio-clipped GMM sensitivity results have a different geometry and are not included in the same-space ranking. The initial global preprocessing fit all 837,951 validated customers before the historical split, so these original-space comparisons inherit that exposure.

| Seed | k | Traditional K-Means-9 S / DB / CH | Traditional GMM-9 S / DB / CH |
|---:|---:|---|---|
| 42 | 2 | 0.3327 / 1.1637 / 922.8 | 0.2946 / 1.6708 / 661.5 |
| 2024 | 2 | 0.3327 / 1.1637 / 922.8 | 0.3308 / 1.1771 / 920.5 |
| 42 | 3 | 0.2215 / 1.6064 / 895.8 | 0.2599 / 1.6333 / 587.9 |
| 2024 | 3 | 0.2216 / 1.6072 / 895.8 | 0.2614 / 1.6331 / 589.9 |

AE-8 and traditional K-Means-8 are close in the common original eight-feature geometry, but assignments vary by seed. Removing `Avg_Spend_Ratio` changes AE memberships substantially at k=2/3 and is a sensitivity analysis, not evidence that the feature should automatically be removed. The four autoencoder inputs come from the historically full-population-scaled matrix; all 42 metric-space rows remain in the comparison ledger/metadata.

## Runtime, artifacts, and limitations

Execution used CPU-only PyTorch 2.13.0+cpu and one Torch thread. The host reported 4.83 GB free physical memory at start; working-set peaks across candidates ranged 425.5–488.9 MiB. AE-9 candidate times were 78.71–84.75 seconds, AE-8 76.22–77.75 seconds, and traditional K-Means-8 0.65–1.36 seconds. Shared network fit time appears in each k-specific candidate runtime.

Models, latent scalers, checkpoints, epoch histories, assignments, segment profiles, and run metadata remain under the existing `artifacts/` layout. The append-only comparison ledger retains its schema and explicit metric-space metadata. PyTorch is an additional inference dependency for AE candidates; the traditional K-Means baselines have lower inference complexity.

The 3,000-customer evaluation sample is repeatedly reused across candidates and therefore is not an independent champion test. The population is dominated by single-transaction customers, and multiple AE segments strongly reflect singleton status. Reconstruction and latent metrics do not establish banking utility or fairness. No champion was selected.

## Verification

| Check | Result |
|---|---|
| Synthetic autoencoder smoke | Passed |
| Real-data smoke | 2 metric-space records completed |
| Comparison | 42/42 metric-space records completed; zero failed/degenerate |
| Checkpoint/scaler/model reload | Predictions reproduced; CustomerID order verified |
| Full project suite at the time | 45 passed |
| Ruff, compileall, `git diff --check` | Passed |

The full suite used the host CPU-PyTorch installation with the existing project site-packages exposed process-locally; the virtual environment was not modified for PyTorch.
