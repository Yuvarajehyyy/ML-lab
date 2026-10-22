"""Phishing Website Detection using Logistic Regression.

This project uses the precomputed website features in dataset_phishing.csv.
It does not extract features from an arbitrary URL entered by a user.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("dataset_phishing.csv")
TARGET_COLUMN = "status"
URL_COLUMN = "url"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "dataset_phishing.csv was not found. Put the CSV in the same folder as main.py."
        )

    data = pd.read_csv(DATASET_PATH)
    print("Dataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())
    print("\nClass distribution:")
    print(data[TARGET_COLUMN].value_counts())

    # Keep the target labels and exclude the raw URL from the input features.
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    y = data[TARGET_COLUMN].map({"legitimate": 0, "phishing": 1})
    valid_rows = y.notna()
    data = data.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    X = data.drop(columns=[TARGET_COLUMN, URL_COLUMN], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Phishing"],
    ).plot()
    plt.title("Phishing Website Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion matrix to confusion_matrix.png")
    plt.show()

    # Demonstration using a row from the CSV. Its features are already provided.
    sample_index = 1 if len(data) > 1 else 0
    sample_features = X.iloc[[sample_index]]
    predicted_label = "Phishing" if model.predict(sample_features)[0] == 1 else "Legitimate"
    actual_label = str(data.iloc[sample_index][TARGET_COLUMN]).capitalize()
    sample_url = data.iloc[sample_index][URL_COLUMN] if URL_COLUMN in data.columns else "Not available"

    print("\nSample prediction from an existing dataset row:")
    print("Dataset URL:", sample_url)
    print("Actual label:", actual_label)
    print("Predicted label:", predicted_label)
    print("\nNote: This demo predicts from precomputed CSV features; it does not scan a live website.")


if __name__ == "__main__":
    main()
