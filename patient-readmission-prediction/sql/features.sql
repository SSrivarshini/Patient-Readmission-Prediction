-- Modelling table: raw columns + engineered features derived in SQL
SELECT
    patient_id,
    age,
    gender,
    bmi,
    smoker,
    insurance_type,
    primary_diagnosis,
    discharge_disposition,
    prior_admissions,
    er_visits_last_year,
    length_of_stay,
    num_medications,
    num_lab_procedures,
    comorbidity_count,
    history_heart_disease,
    history_kidney_disease,
    hba1c,
    systolic_bp,

    -- engineered features
    prior_admissions + er_visits_last_year        AS utilization_score,
    CASE WHEN num_medications >= 10 THEN 1 ELSE 0 END AS polypharmacy,
    CASE WHEN length_of_stay  >= 7  THEN 1 ELSE 0 END AS long_stay,
    CASE WHEN age >= 65             THEN 1 ELSE 0 END AS senior,

    readmitted_30d
FROM patients;
