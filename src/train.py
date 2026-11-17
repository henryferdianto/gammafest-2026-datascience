"""Train and select the best gradient-boosting model with stratified CV."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def candidates():
    return {
        "histgb": HistGradientBoostingClassifier(random_state=42),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
    }


def train(df: pd.DataFrame, target: str = "target"):
    X, y = df.drop(columns=[target]), df[target]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, best_score = None, -1.0
    for name, model in candidates().items():
        score = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
        print(f"{name}: macro-F1 = {score:.4f}")
        if score > best_score:
            best, best_score = model, score
    best.fit(X, y)
    return best, list(X.columns), best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--target", default="target")
    args = ap.parse_args()
    if not Path(args.train).exists():
        from src.sample_data import main as make_sample

        make_sample()
    model, columns, score = train(pd.read_csv(args.train), args.target)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "cv_score": score}, "models/model.joblib")
    print(f"best model saved (CV macro-F1 {score:.4f})")


if __name__ == "__main__":
    main()
