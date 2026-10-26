# Fraud Detection Demo

Streamlit app that scores credit-card transactions with a **supervised XGBoost** model (main decision) and an
**unsupervised autoencoder** (second opinion). Trained and evaluated on the public
[ULB credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud).

**Live demo:** _add your Streamlit link here after deploying_

> Portfolio project on public, anonymised data. Not for real financial decisions.

## Results (held-out test set; thresholds chosen on validation data only)

| model | split | precision | recall | f1 | pr_auc | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | validation | 0.235 | 0.34 | 0.278 | 0.14 | 32 | 104 | 62 | 56547 |
| Isolation Forest | test | 0.248 | 0.379 | 0.3 | 0.233 | 36 | 109 | 59 | 56542 |
| Autoencoder | validation | 0.509 | 0.309 | 0.384 | 0.294 | 29 | 28 | 65 | 56623 |
| Autoencoder | test | 0.524 | 0.347 | 0.418 | 0.421 | 33 | 30 | 62 | 56621 |
| XGBoost (supervised) | validation | 0.945 | 0.734 | 0.826 | 0.797 | 69 | 4 | 25 | 56647 |
| XGBoost (supervised) | test | 0.974 | 0.8 | 0.879 | 0.875 | 76 | 2 | 19 | 56649 |

The test set contains only 95 frauds, so these numbers are noisy. Unsupervised models (Isolation Forest, autoencoder)
are benchmarks; XGBoost uses fraud labels, which is why it scores much higher.

## Method
- Duplicate rows removed before splitting; stratified 60/20/20 train/validation/test split.
- `Time` dropped, `Amount` log-transformed, all features standardised (scaler fitted on training genuine rows only).
- Isolation Forest and an autoencoder trained on genuine transactions only; XGBoost trained on labelled data with
  class weighting and early stopping on validation PR-AUC.
- Decision thresholds picked on the validation set, never on the test set.
- Explanations: XGBoost feature contributions (exact TreeSHAP values) and per-feature autoencoder reconstruction error.
  These explain a score; they are not proof of fraud.

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py`: the app
- `fraud_core.py`: scoring code (identical preprocessing to training)
- `xgb_baseline.json`, `xgb_config.json`: XGBoost model and threshold
- `ae_weights.npz`, `config.json`: autoencoder weights, preprocessing numbers, threshold
- `golden_scores.csv`, `sample_transactions.csv`, `answer_key.csv`, `results.csv`: demo and verification data

## Limitations
Public data from two days in 2013; random (not time-based) split; small number of test frauds; supervised models miss
new fraud patterns; the autoencoder's REVIEW flag is a weak signal (about 2 of 30 REVIEW rows were real frauds).
