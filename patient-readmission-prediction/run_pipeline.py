"""Run everything: generate data (if missing) -> load into SQLite -> train and evaluate.

Run: python run_pipeline.py
"""
from src.build_database import build_db
from src.config import RAW_CSV, TARGET
from src.generate_data import generate
from src.train import main as train

if __name__ == "__main__":
    if not RAW_CSV.exists():
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        data = generate()
        data.to_csv(RAW_CSV, index=False)
        print(f"Generated {len(data):,} synthetic records (readmission rate {data[TARGET].mean():.1%})")
    print(f"Loaded {build_db():,} records into SQLite")
    train()
