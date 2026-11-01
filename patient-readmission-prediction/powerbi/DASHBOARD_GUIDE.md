# Power BI dashboard guide

After running `python run_pipeline.py`, two files are created in `reports/powerbi/`:

| File | Use |
|---|---|
| `readmission_predictions.csv` | One row per test-set patient: demographics, diagnosis, actual outcome, `risk_score`, `risk_band` (Low / Medium / High) |
| `model_metrics.csv` | ROC-AUC, recall and precision for each model |

## Load the data
1. Open Power BI Desktop, **Get data > Text/CSV**, and load both files.
2. In Power Query, check `actual_readmitted` and `predicted_high_risk` are whole numbers.

## Suggested measures (DAX)
```
Patients = COUNTROWS(readmission_predictions)
Actual Readmission Rate = AVERAGE(readmission_predictions[actual_readmitted])
High Risk Patients = CALCULATE([Patients], readmission_predictions[risk_band] = "High")
High Risk Share = DIVIDE([High Risk Patients], [Patients])
Avg Risk Score = AVERAGE(readmission_predictions[risk_score])
```

## Suggested pages
1. **Overview**: cards for Patients, Actual Readmission Rate and High Risk Share; donut chart of `risk_band`; bar chart of readmission rate by `primary_diagnosis`.
2. **Patient risk**: table of High-risk patients sorted by `risk_score`; slicers for age, gender, diagnosis, discharge disposition.
3. **Model performance**: clustered bar chart from `model_metrics.csv` (ROC-AUC, recall, precision by model); insert `reports/figures/roc_curves.png` and `feature_importance.png` as images.

Save your dashboard as `powerbi/Patient_Readmission_Dashboard.pbix` and add a screenshot to the README.
