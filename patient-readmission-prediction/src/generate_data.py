"""Generate a synthetic patient dataset (10,000 records) so the project runs end to end.

No real patient data is included in this repository. To use your own data, replace
data/raw/patients.csv with a file that has the same columns (or adapt sql/features.sql).

Run: python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RAW_CSV, TARGET

DIAGNOSES = ["heart_failure", "copd", "diabetes", "pneumonia", "hypertension", "other"]
DIAGNOSIS_RISK = {
    "heart_failure": 0.60, "copd": 0.50, "diabetes": 0.30,
    "pneumonia": 0.20, "hypertension": 0.0, "other": -0.10,
}
DISPOSITIONS = ["home", "home_health", "skilled_nursing", "rehab"]
DISPOSITION_RISK = {"home": -0.10, "home_health": 0.0, "skilled_nursing": 0.20, "rehab": 0.10}
TARGET_READMISSION_RATE = 0.18


def _find_intercept(z: np.ndarray, target_rate: float) -> float:
    lo, hi = -10.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def generate(n: int = 10_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(62, 15, n), 18, 95).round().astype(int)
    gender = rng.choice(["F", "M"], n)
    bmi = np.clip(rng.normal(28.5, 6, n), 15, 55).round(1)
    smoker = rng.choice(["Y", "N"], n, p=[0.22, 0.78])
    insurance = rng.choice(["medicare", "medicaid", "private", "self_pay"], n, p=[0.45, 0.20, 0.30, 0.05])

    diagnosis = rng.choice(DIAGNOSES, n, p=[0.16, 0.14, 0.20, 0.14, 0.20, 0.16])
    disposition = rng.choice(DISPOSITIONS, n, p=[0.55, 0.20, 0.15, 0.10])

    prior_adm = rng.poisson(1.1, n)
    er_visits = rng.poisson(0.9, n)
    los = np.clip(rng.gamma(2.2, 2.0, n).round().astype(int), 1, 30)
    meds = np.clip(rng.poisson(9, n), 1, 40)
    labs = np.clip(rng.normal(43, 19, n), 1, 120).round().astype(int)
    comorb = np.clip(rng.poisson(1.8, n), 0, 8)

    hist_heart = rng.binomial(1, 1 / (1 + np.exp(-(age - 70) / 10)))
    hist_kidney = rng.binomial(1, 0.12 + 0.002 * np.clip(age - 40, 0, None))
    hba1c = np.clip(rng.normal(6.4, 1.3, n), 4.0, 14.0).round(1)
    sbp = np.clip(rng.normal(132, 18, n), 80, 220).round()

    signal = (
        0.02 * (age - 62) + 0.30 * prior_adm + 0.20 * er_visits + 0.06 * (los - 5)
        + 0.03 * (meds - 9) + 0.01 * (labs - 43) + 0.15 * comorb
        + 0.40 * hist_heart + 0.35 * hist_kidney + 0.12 * (hba1c - 6.4)
        + 0.01 * (bmi - 28) + 0.25 * (smoker == "Y") + 0.30 * (insurance == "self_pay")
        + np.array([DIAGNOSIS_RISK[d] for d in diagnosis])
        + np.array([DISPOSITION_RISK[d] for d in disposition])
    )
    z = 2.0 * signal + rng.normal(0, 0.5, n)  # amplify signal, add random noise
    intercept = _find_intercept(z, TARGET_READMISSION_RATE)
    readmitted = rng.binomial(1, 1 / (1 + np.exp(-(z + intercept))))

    df = pd.DataFrame({
        "patient_id": [f"P{100000 + i}" for i in range(n)],
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoker": smoker,
        "insurance_type": insurance,
        "primary_diagnosis": diagnosis,
        "discharge_disposition": disposition,
        "prior_admissions": prior_adm,
        "er_visits_last_year": er_visits,
        "length_of_stay": los,
        "num_medications": meds,
        "num_lab_procedures": labs,
        "comorbidity_count": comorb,
        "history_heart_disease": hist_heart,
        "history_kidney_disease": hist_kidney,
        "hba1c": hba1c,
        "systolic_bp": sbp,
        TARGET: readmitted,
    })

    # Realistic missing values in clinical measurements
    for col in ["bmi", "hba1c", "systolic_bp"]:
        df.loc[rng.random(n) < 0.06, col] = np.nan
    return df


if __name__ == "__main__":
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(data):,} records to {RAW_CSV} (readmission rate {data[TARGET].mean():.1%})")
