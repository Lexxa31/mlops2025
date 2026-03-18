import pickle
import pandas as pd
import yaml
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")
    n_rows = df.shape[0]  # количество строк в данных

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total rows in dataset: {n_rows}")

    # Сохраняем метрики в JSON
    metrics = {
        "accuracy": accuracy,
        "n_rows": n_rows
    }

    # Гарантируем существование папки (хотя по условию она уже создана)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    evaluate_model()