# Patient Readmission Prediction

A machine learning project that predicts which patients are likely to be **readmitted to hospital within 30 days** of discharge, so that high-risk patients can be flagged early for follow-up care.

It covers the full workflow: SQL-based data preparation, preprocessing and feature engineering, training and comparing **Random Forest** and **Logistic Regression** models, threshold tuning to improve recall for high-risk patients, and a **Power BI** dashboard for presenting insights.

> **Note on data:** no real patient data is included. The repository generates a **synthetic dataset of 10,000 patient records** (`src/generate_data.py`) so the whole pipeline can be run end to end. To use your own data, replace `data/raw/patients.csv` (same columns) and re-run. Metrics from the synthetic data are for demonstration only.

## What it does

- **Data preparation with SQL**: records are loaded into SQLite and the modelling table is built with `sql/features.sql`, including engineered features (utilisation score, polypharmacy, long stay, senior)
- **Preprocessing**: median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns, all inside a scikit-learn `Pipeline`
- **Modelling**: Logistic Regression (baseline and class-balanced) and Random Forest, compared on a stratified hold-out test set
- **Recall for high-risk patients**: the decision threshold of the best model is tuned on out-of-fold predictions (F2 score, which favours recall) and compared with the default 0.5 baseline
- **Explainability**: ROC curves and Random Forest feature importance saved as images
- **Power BI ready**: predictions with risk bands and model metrics are exported as CSV files, with a dashboard guide in `powerbi/`

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, SQL (SQLite), Random Forest, Logistic Regression, Power BI

## Project Structure

```
patient-readmission-prediction/
├── data/raw/                 # patients.csv is generated here
├── sql/
│   ├── features.sql          # modelling table + SQL feature engineering
│   └── eda_queries.sql       # example analysis queries
├── src/
│   ├── generate_data.py      # synthetic 10,000-record dataset
│   ├── build_database.py     # CSV -> SQLite
│   ├── data_loader.py        # runs features.sql
│   ├── preprocess.py         # imputation, encoding, scaling
│   └── train.py              # train, evaluate, tune threshold, export
├── powerbi/DASHBOARD_GUIDE.md
├── tests/test_pipeline.py
├── run_pipeline.py           # runs everything
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/SSrivarshini/patient-readmission-prediction.git
cd patient-readmission-prediction
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

This generates the data, loads it into SQLite, trains the models and writes:

| Output | Contents |
|---|---|
| `reports/metrics.json` | ROC-AUC, recall, precision per model, tuned threshold and recall improvement |
| `reports/figures/` | `roc_curves.png`, `feature_importance.png` |
| `reports/powerbi/` | `readmission_predictions.csv`, `model_metrics.csv` for Power BI |
| `models/readmission_model.joblib` | Best model and its tuned threshold |

Run the tests with `pytest`.

## Evaluation approach

- **ROC-AUC** measures how well each model ranks patients by risk
- **Recall** for readmitted patients measures how many high-risk patients are caught, which matters most in a clinical follow-up setting
- The tuned model is compared against the default Logistic Regression at threshold 0.5, and the change in recall is reported in `reports/metrics.json`

## Power BI Dashboard

See [`powerbi/DASHBOARD_GUIDE.md`](powerbi/DASHBOARD_GUIDE.md) for the data files, DAX measures and page layout used to present patient insights, model results and performance metrics.

## Future Improvements

- Cross-validated hyperparameter search for Random Forest and gradient boosting models
- Probability calibration and SHAP explanations for individual predictions
- A small API or app to score new patients
- Fairness checks across age and gender groups

## Author

**Srivarshini S**: [GitHub](https://github.com/SSrivarshini)

## License

MIT
