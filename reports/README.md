# Reports

Reviewed technical reports for data quality, feature behavior, model comparisons, validation, and deployment readiness.

## Track A

- [Feature distribution diagnostics](track_a_feature_diagnostics.md) — transaction history, feature tails, and ratio sensitivity.
- [Traditional clustering baselines](track_a_traditional_clustering_baselines.md) — K-Means and diagonal GMM results in the historical globally transformed feature matrix.
- [Autoencoder representation evaluation](track_a_autoencoder_evaluation.md) — matched historical AE and eight-feature ablation results, with latent metrics kept separate.
- [Behavioral robustness review](track_a_behavioral_robustness_review.md) — evaluation reuse, transaction-history effects, ratio sensitivity, and repeat-only analysis.
- [Train-fitted repeat-customer segmentation](track_a_train_fitted_segmentation.md) — raw-aggregate train-fitted preprocessing, corrected clipping diagnostics, and candidate results.
- [Matched autoencoder comparison](track_a_matched_model_comparison.md) — fixed-architecture AE vs. train-fitted K-Means/GMM baselines.
- [Model readiness and review framework](track_a_model_readiness.md) — finalist artifacts, validation feasibility, fairness limitations, and inference contract.
- [Segment interpretation review](track_a_segment_interpretation_review.md) — candidate profiles and demographic representation for downstream review.
- [Independent validation protocol](track_a_independent_validation_protocol.md) — manifest exposure and prospective validation requirements.

Numerical run IDs, exact artifact paths, and hashes remain in the comparison ledger and versioned machine-readable audit directories. The reports use descriptive names rather than the original development milestones.
