"""
train.py  -  Fake vs Genuine account detection: final model training script.

What this script does (same workflow as the notebook, only the winning model):
  1. Loads fake_account_detection.csv
  2. Removes duplicate rows
  3. Drops the identifier columns (account_id, signup_date)
  4. Makes a stratified 80/20 train-test split (random_state=42)
  5. Builds the preprocessing + model Pipeline:
        numeric columns     -> median imputer -> StandardScaler
        categorical columns -> most-frequent imputer -> OneHotEncoder(handle_unknown='ignore')
        classifier          -> LogisticRegression(max_iter=1000)
  6. Tunes C with GridSearchCV (5-fold stratified CV, scoring = F1 of the 'Fake' class)
     using ONLY the training data (the notebook found C = 1)
  7. Evaluates ONCE on the untouched test set
  8. Saves everything the app needs into model.pkl

app.py also imports train_model() from here to rebuild model.pkl automatically when the
saved file was made with a different scikit-learn version.

Run:   python train.py
Optional:  python train.py --data path/to/file.csv --output path/to/model.pkl
"""

import argparse
import builtins
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "fake_account_detection.csv"
DEFAULT_MODEL = BASE_DIR / "model.pkl"
builtins_print = builtins.print

TARGET = "account_status"
POSITIVE_LABEL = "Fake"                      # the class we care most about finding
DROP_COLUMNS = ["account_id", "signup_date"]  # identifiers, not useful as features
BINARY_FEATURES = ["profile_photo_present"]  # 0/1 column (notebook treats it as categorical)
RANDOM_STATE = 42
TEST_SIZE = 0.20
PARAM_GRID = {"classifier__C": [0.1, 1, 10]}  # same grid as the notebook


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. "
            "Put fake_account_detection.csv in the same folder as train.py "
            "(or use --data path/to/file.csv)."
        )
    df = pd.read_csv(path)
    needed = [TARGET] + BINARY_FEATURES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"these required columns are missing from the CSV: {missing}")
    return df


def build_pipeline(numerical_features, categorical_features) -> Pipeline:
    numerical_preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_preprocessing, numerical_features),
        ("cat", categorical_preprocessing, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def to_python(value):
    """Convert numpy scalars to plain Python values so the pickle stays simple."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_feature_metadata(X: pd.DataFrame, numerical_features):
    """Describe every input column so app.py can build its form from real data."""
    features = []
    for col in X.columns:
        series = X[col].dropna()
        if col in numerical_features:
            features.append({
                "name": col,
                "kind": "numeric",
                "min": to_python(series.min()),
                "max": to_python(series.max()),
                "median": to_python(series.median()),
                "is_integer": bool((series % 1 == 0).all()),
            })
        elif col in BINARY_FEATURES:
            features.append({
                "name": col,
                "kind": "binary",
                "options": [0.0, 1.0],
                "default": to_python(float(series.mode().iloc[0])),
            })
        else:
            features.append({
                "name": col,
                "kind": "category",
                "options": sorted(series.astype(str).unique().tolist()),
                "default": str(series.mode().iloc[0]),
            })
    return features


def train_model(data_path=DEFAULT_DATA, output_path=DEFAULT_MODEL, verbose=True):
    """Train the final model, save model.pkl and return the saved bundle.

    verbose=True prints the full report (used by `python train.py`);
    verbose=False is silent (used by app.py when it has to rebuild model.pkl).
    """
    def print(*a, **k):  # noqa: A001  - silent when verbose=False
        if verbose:
            builtins_print(*a, **k)

    # 1. Load -------------------------------------------------------------
    df = load_data(Path(data_path))
    print("Dataset shape:", df.shape)

    # 2. Duplicates / unusable rows (same as the notebook) -----------------
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {duplicates} -> shape {df.shape}")

    if df[TARGET].isna().any():
        print("Rows with a missing target are dropped:", int(df[TARGET].isna().sum()))
        df = df.dropna(subset=[TARGET])
    labels = sorted(df[TARGET].unique())
    if POSITIVE_LABEL not in labels:
        raise ValueError(f"target column has labels {labels}, expected '{POSITIVE_LABEL}' among them.")
    print("Target distribution:\n", df[TARGET].value_counts().to_string())

    # 3. X / y -------------------------------------------------------------
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
    y = df[TARGET]

    # 4. Stratified split (before any fitting, so there is no leakage) ------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Training data:", X_train.shape, " Testing data:", X_test.shape)

    # 5. Column types (same rule as the notebook) --------------------------
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    for col in BINARY_FEATURES:
        if col in numerical_features:
            numerical_features.remove(col)
    categorical_features = [c for c in X_train.columns if c not in numerical_features]
    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features), "(includes", BINARY_FEATURES, ")")

    # 6. Tune on the training data only ------------------------------------
    pipeline = build_pipeline(numerical_features, categorical_features)
    fake_f1_scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid=PARAM_GRID, scoring=fake_f1_scorer, cv=cv, n_jobs=-1)
    print("\nRunning GridSearchCV (training data only)...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print("Best cross-validated F1 (Fake):", round(grid.best_score_, 4))

    # 7. Evaluate once on the held-out test set -----------------------------
    y_pred = best_model.predict(X_test)
    classes = list(best_model.classes_)
    fake_prob = best_model.predict_proba(X_test)[:, classes.index(POSITIVE_LABEL)]
    is_fake = (y_test == POSITIVE_LABEL).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        "recall": recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        "f1": f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        "roc_auc": roc_auc_score(is_fake, fake_prob),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[POSITIVE_LABEL] + [c for c in classes if c != POSITIVE_LABEL])

    print("\n=== Test-set results (Fake is the positive class) ===")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted; order:",
          [POSITIVE_LABEL] + [c for c in classes if c != POSITIVE_LABEL], ")")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save everything the app needs in ONE file ---------------------------
    bundle = {
        "pipeline": best_model,                       # fitted preprocessing + classifier
        "feature_order": X.columns.tolist(),          # exact column order used in training
        "features": build_feature_metadata(X, numerical_features),
        "classes": classes,
        "positive_label": POSITIVE_LABEL,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "best_params": {k: to_python(v) for k, v in grid.best_params_.items()},
        "cv_f1": float(grid.best_score_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, output_path)
    print(f"\nSaved model to: {output_path}")

    # 9. Reload and test right away ------------------------------------------
    loaded = joblib.load(output_path)
    sample = X_test.iloc[[0]][loaded["feature_order"]]
    print("Reload check -> sample prediction:", loaded["pipeline"].predict(sample)[0],
          "| actual:", y_test.iloc[0])
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train the Fake vs Genuine account model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to the CSV file")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="where to save model.pkl")
    args = parser.parse_args()
    try:
        train_model(args.data, args.output, verbose=True)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
