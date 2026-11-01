import numpy as np

from src.generate_data import generate
from src.preprocess import build_preprocessor
from src.train import best_threshold


def test_generated_data_shape_and_rate():
    df = generate(n=1000)
    assert len(df) == 1000
    assert 0.05 < df["readmitted_30d"].mean() < 0.40


def test_preprocessor_removes_missing_values():
    df = generate(n=500)
    X = df.drop(columns=["readmitted_30d", "patient_id"])
    assert X.isna().any().any()  # data really has missing values
    Xt = build_preprocessor(X).fit_transform(X)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(dense).any()


def test_best_threshold_is_a_probability():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.6, 0, 1)
    assert 0.0 <= best_threshold(y, proba) <= 1.0
