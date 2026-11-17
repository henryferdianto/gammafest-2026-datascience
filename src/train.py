"""Train and ensemble gradient-boosting models with stratified CV."""
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="target")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    X = pd.get_dummies(df.drop(columns=[args.target]), drop_first=True)
    y = df[args.target]

    models = {
        "lgbm": LGBMClassifier(n_estimators=600, learning_rate=0.03, random_state=42),
        "xgb": XGBClassifier(n_estimators=600, learning_rate=0.03, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
        print(f"{name}: {score:.4f}")
        model.fit(X, y)
        joblib.dump(model, f"models/{name}.joblib")


if __name__ == "__main__":
    main()
