import pandas as pd

from src.predict import predict
from src.sample_data import make
from src.train import train


def test_train_and_predict():
    train_df, test_df = make(600)
    model, columns, score = train(train_df)
    assert score > 0.4
    sub = predict({"model": model, "columns": columns}, test_df)
    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(test_df)
