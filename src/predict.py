"""Generate a submission with the averaged ensemble."""
import argparse
import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.test)
    X = pd.get_dummies(df, drop_first=True)
    models = [joblib.load(f"models/{m}.joblib") for m in ("lgbm", "xgb")]
    proba = sum(m.predict_proba(X) for m in models) / len(models)
    sub = pd.DataFrame({"id": df.iloc[:, 0], "prediction": proba.argmax(axis=1)})
    sub.to_csv("submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
