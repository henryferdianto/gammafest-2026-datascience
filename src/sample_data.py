"""Generate a synthetic multi-class tabular competition dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 8


def make(n: int = 2000, n_classes: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, N_FEATURES))
    weights = rng.normal(0, 1, (N_FEATURES, n_classes))
    logits = X @ weights + rng.normal(0, 0.3, (n, n_classes))
    y = logits.argmax(axis=1)
    train = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    train["target"] = y
    test = pd.DataFrame(rng.normal(0, 1, (n // 4, N_FEATURES)),
                        columns=[f"f{i}" for i in range(N_FEATURES)])
    test.insert(0, "id", range(len(test)))
    return train, test


def main() -> Path:
    import pathlib

    train, test = make()
    for name, df in (("train", train), ("test", test)):
        out = pathlib.Path(f"data/raw/{name}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {name} -> {out}")
    return pathlib.Path("data/raw/train.csv")


if __name__ == "__main__":
    main()
