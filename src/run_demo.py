import pandas as pd

from src.predict import predict
from src.sample_data import make
from src.train import train


def main():
    train_df, test_df = make()
    model, columns, score = train(train_df)
    sub = predict({"model": model, "columns": columns}, test_df)
    print(f"CV macro-F1={score:.4f}, submission rows={len(sub)}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
