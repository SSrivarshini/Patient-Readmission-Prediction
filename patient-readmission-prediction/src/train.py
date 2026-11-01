"""Train and evaluate Logistic Regression and Random Forest readmission models.

Run: python -m src.train
Outputs: models/, reports/metrics.json, reports/figures/, reports/powerbi/
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR, ID_COL, MODELS_DIR, POWERBI_DIR, RANDOM_STATE, REPORTS_DIR, TARGET,
)
from src.data_loader import load_features
from src.preprocess import build_preprocessor


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold that maximises F-beta (beta=2 favours recall, i.e. catching high-risk patients)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f_beta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    return float(thresholds[np.argmax(f_beta)])


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "threshold": round(float(threshold), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
    }


def make_models(X) -> dict:
    def pipe(model):
        return Pipeline([("prep", build_preprocessor(X)), ("model", model)])

    return {
        "logistic_regression_baseline": pipe(LogisticRegression(max_iter=1000)),
        "logistic_regression_balanced": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    }


def save_figures(fitted: dict, X_test, y_test) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    rf = fitted["random_forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(rf.named_steps["model"].feature_importances_, index=names).nlargest(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    importances[::-1].plot.barh(ax=ax)
    ax.set_title("Random Forest: top 15 features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_features()
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"{len(df):,} records | readmission rate {y.mean():.1%} | train {len(X_train):,} / test {len(X_test):,}")

    fitted, results = {}, {}
    for name, model in make_models(X).items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = evaluate(y_test, model.predict_proba(X_test)[:, 1])
        print(f"{name:32s} {results[name]}")

    # Pick the best model by ROC-AUC, then tune its decision threshold for recall
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    oof = cross_val_predict(clone(best), X_train, y_train, cv=5, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)
    test_proba = best.predict_proba(X_test)[:, 1]
    tuned = evaluate(y_test, test_proba, threshold)

    baseline_recall = results["logistic_regression_baseline"]["recall"]
    improvement = {
        "baseline_model": "logistic_regression_baseline (threshold 0.5)",
        "tuned_model": f"{best_name} (tuned threshold)",
        "baseline_recall": baseline_recall,
        "tuned_recall": tuned["recall"],
        "absolute_change": round(tuned["recall"] - baseline_recall, 4),
    }
    print(f"\nBest model: {best_name}\nTuned: {tuned}\nHigh-risk recall vs baseline: {improvement}")

    # Save model, metrics, figures
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "threshold": threshold}, MODELS_DIR / "readmission_model.joblib")
    metrics = {"models": results, "best_model": best_name, "tuned": tuned, "high_risk_recall": improvement}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_figures(fitted, X_test, y_test)

    # Files for the Power BI dashboard
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    out = df.loc[X_test.index, [
        ID_COL, "age", "gender", "primary_diagnosis", "discharge_disposition",
        "insurance_type", "length_of_stay", "utilization_score",
    ]].copy()
    out["actual_readmitted"] = y_test.values
    out["risk_score"] = test_proba.round(4)
    out["predicted_high_risk"] = (test_proba >= threshold).astype(int)
    out["risk_band"] = np.where(test_proba >= threshold, "High", np.where(test_proba >= threshold / 2, "Medium", "Low"))
    out.to_csv(POWERBI_DIR / "readmission_predictions.csv", index=False)

    rows = [{"model": k, **v} for k, v in results.items()] + [{"model": f"{best_name} (tuned)", **tuned}]
    pd.DataFrame(rows).to_csv(POWERBI_DIR / "model_metrics.csv", index=False)
    print(f"\nSaved model, metrics, figures and Power BI files under {REPORTS_DIR.parent}")


if __name__ == "__main__":
    main()
