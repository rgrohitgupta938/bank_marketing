# src/train.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.fe import load_bank_data, build_preprocessor, get_feature_names
from src.utils import (
    make_run_id,
    ensure_dir,
    setup_logger,
    write_json,
    write_latest_run,
)
from src.evaluate import evaluate_model


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/bank-full.csv"
ARTIFACTS_ROOT = "artifacts"
MODELS_DIR = "models"
LOG_FILE = "logs/train.log"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def to_dense(X):
    """
    Convert X to a dense numpy array if it is sparse-like.
    Works for:
      - numpy.ndarray (returns as-is)
      - scipy sparse matrices (has .toarray())
      - anything that supports np.asarray
    """
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def main():
    run_id = make_run_id("bank")
    run_dir = os.path.join(ARTIFACTS_ROOT, "runs", run_id)
    ensure_dir(run_dir)
    ensure_dir(MODELS_DIR)

    logger = setup_logger(LOG_FILE)
    logger.info(f"Starting training run: {run_id}")

    # -----------------------------
    # Load data
    # -----------------------------
    logger.info("Loading dataset...")
    X, y, df = load_bank_data(DATA_PATH)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Train size: {X_train.shape}")
    logger.info(f"Test size: {X_test.shape}")

    # -----------------------------
    # Build + Fit Preprocessor
    # -----------------------------
    logger.info("Building preprocessing pipeline...")
    preprocessor = build_preprocessor()

    logger.info("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Processed train type: {type(X_train_processed)}")
    logger.info(f"Processed test type: {type(X_test_processed)}")

    # Save feature names (best effort)
    feature_names = get_feature_names(preprocessor)
    write_json(
        os.path.join(run_dir, "feature_columns.json"),
        {"feature_names": feature_names},
    )

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

    # -----------------------------
    # Define Models
    # -----------------------------
    logger.info("Initializing models...")

    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    models = {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }

    metrics_list = []

    # -----------------------------
    # Train + Evaluate Each Model
    # -----------------------------
    for name, model in models.items():
        logger.info(f"Training model: {name}")
        start_time = time.time()

        if name == "naive_bayes":
            Xtr = to_dense(X_train_processed)
            Xte = to_dense(X_test_processed)

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]
        else:
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1]

        train_time = time.time() - start_time
        logger.info(f"{name} training time: {train_time:.2f} seconds")

        metrics = evaluate_model(
            model_name=name,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            run_dir=run_dir,
        )
        metrics["train_time_sec"] = round(train_time, 3)
        metrics_list.append(metrics)

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

        logger.info(f"{name} saved to {model_path}")
        logger.info(f"{name} metrics: {metrics}")

    # -----------------------------
    # Save Metrics Comparison Table
    # -----------------------------
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(run_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics comparison saved to {metrics_path}")

    # -----------------------------
    # Save Run Config
    # -----------------------------
    run_config = {
        "dataset_path": DATA_PATH,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "models": list(models.keys()),
    }
    write_json(os.path.join(run_dir, "run_config.json"), run_config)
    write_latest_run(ARTIFACTS_ROOT, run_id)

    logger.info(f"Training completed successfully for run: {run_id}")


if __name__ == "__main__":
    main()
