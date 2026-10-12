# Track A Matched Autoencoder and Traditional Clustering Comparison

**Run date:** 2026-10-11
**Population:** 2+ transaction customers
**Decision:** no champion selected; the autoencoder did not show a consistent advantage over the train-fitted K-Means baselines.

## Executive summary

The fixed nine-feature autoencoder completed for seeds 42 and 2024 and produced four latent K-Means candidates (k=2/3). It used the same 50,000-row training manifests, 10,000-row evaluation set, and saved nine-feature train-fitted preprocessors as the traditional baseline comparison.

On the common original nine-feature train-fitted space, AE scores were below K-Means at both k values and both seeds. AE-to-baseline ARIs were near zero, and AE cross-seed ARI was only 0.2097 at k=2 and 0.2566 at k=3. Latent-space silhouettes were higher, but these describe a different geometry and are not a direct ranking. The AE partitions were weakly stable and did not show added value.

## Data, manifests, and fit boundaries

| Split | Rows | Ordered CustomerID SHA-256 |
|---|---:|---|
| Training, seed 42 | 50,000 | `42a4a9b9f170ed2ff3f3cd84d9b06a88efdae3465226f66d0ca0208fe962ac87` |
| Training, seed 2024 | 50,000 | `830bed041f9031cb34e6afcd2f6b80a8c4573baa530ddd0c38159387aa52e164` |
| Shared held-out evaluation | 10,000 | `1430463f2074d93caea16c49f07cc3209d399bad51bc8fba410ceff33112ffd4` |

Training and evaluation IDs are disjoint. The same ordered holdout was used by both seeds and matches the saved traditional candidate assignments. Internal AE splits contain 45,000 training-manifest IDs and 5,000 internal-validation IDs per seed.

The saved train-fitted nine-feature StandardScaler was reused exactly. It was fit on all 50,000 training-manifest rows, including the 5,000 IDs later used for internal AE validation; those validation rows did not enter gradient updates, but they did contribute to the unsupervised scaler statistics. No external evaluation IDs entered early stopping, AE fitting, latent scaling, or K-Means fitting. The initial global scaler was fit on all 837,951 validated customers and is not used by these candidates.

## Model configuration and training

The only architecture was 9→32→16→4→16→32→9, with ReLU hidden layers and linear bottleneck/output. CPU PyTorch 2.13.0+cpu, AdamW (`learning_rate=0.001`, `weight_decay=0.0001`), SmoothL1/Huber (`beta=1`), batch size 512, maximum 100 epochs, and early-stopping patience 10 were used. Best validation checkpoints were restored. Both seeds reached the 100-epoch cap without early stopping.

| Seed | Internal train / validation | Best epoch | Best train Huber | Best validation Huber | AE fit time | AE peak working set |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 45,000 / 5,000 | 100 | 0.0532713 | 0.0549838 | 34.58 s | 475.0 MiB |
| 2024 | 45,000 / 5,000 | 100 | 0.0199853 | 0.0182865 | 31.39 s | 480.3 MiB |

The latent StandardScaler was fit on all 50,000 embeddings per seed. Reloaded scaler fit counts/hashes and model predictions were checked against the saved artifacts and assignment files.

## Held-out AE metrics

Original-space metrics are evaluated in the same seed-specific train-fitted nine-feature matrix as the traditional baseline. Latent metrics use the separately scaled four-dimensional embedding. S/DB/CH denote silhouette / Davies–Bouldin / Calinski–Harabasz; silhouette uses a deterministic 3,000-row sample and DB/CH use all 10,000 evaluation rows.

| Seed | k | Original 9-feature S / DB / CH | Latent 4D S / DB / CH | Membership counts | Minimum segment | Cross-seed ARI | Candidate fit seconds* | Peak working set | Reloaded inference / 10,000 |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 42 | 2 | 0.1087 / 3.0314 / 790.5 | 0.3545 / 1.1687 / 5,083.2 | 6,847 / 3,153 | 31.53% | 0.2097 | 35.01 | 475.9 MiB | 2.55 ms |
| 42 | 3 | 0.0851 / 2.8244 / 729.2 | 0.3264 / 1.0723 / 5,991.0 | 4,140 / 2,789 / 3,071 | 27.89% | 0.2566 | 34.85 | 474.9 MiB | 2.82 ms |
| 2024 | 2 | 0.1169 / 2.8328 / 1,005.5 | 0.4199 / 0.9908 / 9,070.2 | 4,949 / 5,051 | 49.49% | 0.2097 | 31.53 | 480.3 MiB | 2.66 ms |
| 2024 | 3 | 0.1111 / 2.5819 / 816.0 | 0.4152 / 0.8490 / 7,811.7 | 2,694 / 2,640 / 4,666 | 26.40% | 0.2566 | 31.72 | 483.5 MiB | 4.35 ms |

`*` Candidate time is AE fit (shared across k for a seed) plus the corresponding K-Means fit; do not sum k-specific times as independent AE fits. Latent scores are representation diagnostics and are not directly comparable to original-space scores or another seed's independently fitted latent geometry.

## Matched traditional baselines

The unclipped nine-feature K-Means and diagonal GMM are the direct comparison because they use the same 2+ population, ordered 10,000 evaluation IDs, and seed-specific train-fitted feature matrices.

| Seed | k | AE S / DB / CH | K-Means S / DB / CH | GMM S / DB / CH | AE vs K-Means ARI | AE vs GMM ARI |
|---:|---:|---|---|---|---:|---:|
| 42 | 2 | 0.1087 / 3.0314 / 790.5 | 0.1927 / 1.7879 / 2,695.6 | 0.1386 / 5.0446 / 174.1 | 0.0082 | 0.0139 |
| 2024 | 2 | 0.1169 / 2.8328 / 1,005.5 | 0.1924 / 1.7909 / 2,593.1 | 0.1389 / 5.0493 / 183.3 | 0.0004 | -0.0001 |
| 42 | 3 | 0.0851 / 2.8244 / 729.2 | 0.2092 / 1.6128 / 2,165.8 | 0.1353 / 3.7519 / 723.6 | 0.0148 | 0.0131 |
| 2024 | 3 | 0.1111 / 2.5819 / 816.0 | 0.2101 / 1.6119 / 2,084.4 | 0.1464 / 3.5270 / 447.4 | 0.0003 | -0.0037 |

| Family | k=2 cross-seed ARI | k=3 cross-seed ARI |
|---|---:|---:|
| Autoencoder + latent K-Means | 0.2097 | 0.2566 |
| Train-fitted nine-feature K-Means baseline | 0.9908 | 0.9923 |
| Train-fitted nine-feature diagonal GMM baseline | 0.9303 | 0.9505 |

The ratio-clipped GMM is a distinct sensitivity alternative and is not combined with the unclipped same-space table. Its k=2 mean silhouette/DB/CH are 0.1600/3.1161/706.1 and three-seed ARI is 0.9904 (minimum 0.9856). The corrected clipping audit found 101/106/101 held-out ratio values clipped for seeds 42/2024/123. Those values supersede the historical zero-count ledger parameter; the saved models and metrics are preserved.

Historical autoencoder scores from the original global-scaling/all-customer experiment are not treated as a matched comparator. Its population and transform differ from the train-fitted repeat-customer setup.

## Reconstruction and behavioral profiles

Per-feature reconstruction MAE/RMSE for both internal train and validation, in transformed feature units, is in the versioned autoencoder diagnostic output. The largest values include AE-9 seed-42 validation transaction-count RMSE 1.024 and ratio RMSE 0.540; seed-2024 ratio RMSE is 0.542. Reconstruction scores do not indicate cluster quality.

On the shared holdout, AE k=2 clusters have median transaction count 2 and similar median balances (about 29.1–32.1k INR) and total spend (about 1.35–1.69k INR); the seed-42 size split changes from 68.5/31.5 to 49.5/50.5 for seed 2024. At k=3, most clusters remain dominated by exactly-two-transaction customers and median spend/balance differences are modest. Their cross-seed ARIs show these profiles do not map to stable segments.

## Deep-learning value assessment and candidate disposition

The AE latent geometry can have stronger silhouette/DB values than the original feature space, but those metrics are not comparable across geometries. In the shared original space, AE silhouettes are below K-Means for both k values and seeds; AE-to-K-Means/GMM assignment ARIs are approximately zero; AE cross-seed ARIs are low. Training adds 31–35 seconds per seed and a CPU PyTorch inference dependency, without demonstrated stable behavioral value. No AE candidate advances to the proposed finalist shortlist.

The retained review candidates are two traditional K-Means geometries (k=2 and k=3) plus the ratio-clipped GMM k=2 sensitivity alternative. The shortlist and validation gates are in [`track_a_model_readiness.md`](track_a_model_readiness.md); no champion is chosen.

## Artifacts and verification

- Model and training code: `src/customer_segmentation/models/autoencoder.py` and `src/customer_segmentation/models/track_a_autoencoder_experiments.py`
- Comparison configuration: `configs/models/track_a_autoencoder.yaml`
- Candidate models, checkpoints, latent scalers: `artifacts/models/autoencoder/`
- Epoch traces, internal split manifests, assignments, profiles, and metadata: `artifacts/experiments/autoencoder_history/`, `artifacts/experiments/samples/`, `artifacts/segments/candidates/`, `artifacts/experiments/segment_profiles/`, and `artifacts/experiments/run_metadata/`
- Comparison ledger: `artifacts/experiments/comparison_board.csv`; explicit metric-space metadata distinguishes original and latent geometries.

The comparison recorded 42 completed metric-space rows, plus two real-data smoke rows. Checkpoint/scaler/K-Means reloads reproduced predictions. At the original evaluation run, 45 project tests passed; the current full-suite result is recorded in the model-readiness report. DEC/IDEC remains deferred pending an independent validation and clearer behavioral/stability evidence.
