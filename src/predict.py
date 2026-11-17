"""Generate a submission with the saved model."""
import argparse
import joblib
import pandas as pd


def predict(payload: dict, test: pd.DataFrame) -> pd.DataFrame:
    model = payload["model"]
    X = test.drop(columns=["id"], errors="ignore").reindex(
        columns=payload["columns"], fill_value=0
    )
    return pd.DataFrame({"id": test["id"], "prediction": model.predict(X)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--model", default="models/model.joblib")
    args = ap.parse_args()
    payload = joblib.load(args.model)
    sub = predict(payload, pd.read_csv(args.test))
    sub.to_csv("submission.csv", index=False)
    print("wrote submission.csv", sub.shape)


if __name__ == "__main__":
    main()
