-- Example analysis queries (run against data/patients.db, e.g. with DB Browser for SQLite)

-- 1. Overall 30-day readmission rate
SELECT ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct FROM patients;

-- 2. Readmission rate by primary diagnosis
SELECT primary_diagnosis,
       COUNT(*)                              AS patients,
       ROUND(AVG(readmitted_30d) * 100, 1)   AS readmission_rate_pct
FROM patients
GROUP BY primary_diagnosis
ORDER BY readmission_rate_pct DESC;

-- 3. Readmission rate by prior admissions
SELECT prior_admissions, COUNT(*) AS patients, ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM patients
GROUP BY prior_admissions
ORDER BY prior_admissions;

-- 4. Missing values per clinical column
SELECT SUM(bmi IS NULL) AS missing_bmi,
       SUM(hba1c IS NULL) AS missing_hba1c,
       SUM(systolic_bp IS NULL) AS missing_systolic_bp
FROM patients;
