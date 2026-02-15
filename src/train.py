# src/train.py

import os
import time
import joblib
import boto3
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
from src.cloud import upload_dir_to_s3


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/bank-full.csv"
ARTIFACTS_ROOT = "artifacts"
MODELS_ROOT = "models"
LOG_FILE = "logs/train.log"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# S3 CONFIG (ENV-DRIVEN)
# -----------------------------
UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "0") == "1"
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "bank-marketing")  # e.g. "bank-marketing"


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

    # Per-run models directory: models/<run_id>/
    run_models_dir = os.path.join(MODELS_ROOT, run_id)
    ensure_dir(run_models_dir)

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

    # Save preprocessor (LOCAL, per-run)
    preprocessor_path = os.path.join(run_models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

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

        # Save model (LOCAL, per-run)
        model_path = os.path.join(run_models_dir, f"{name}.pkl")
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
        "run_id": run_id,
        "run_dir": run_dir,
        "run_models_dir": run_models_dir,
    }
    write_json(os.path.join(run_dir, "run_config.json"), run_config)

    # Keeps whatever your existing utility does (fine to keep)
    write_latest_run(ARTIFACTS_ROOT, run_id)

    # Streamlit-friendly pointer
    latest_json_path = os.path.join(ARTIFACTS_ROOT, "latest_run.json")
    write_json(latest_json_path, {"run_id": run_id})
    logger.info(f"Latest run pointer saved to {latest_json_path}")

    # -----------------------------
    # Optional: Upload artifacts + THIS RUN's models + latest_run.json to S3
    # -----------------------------
    if UPLOAD_TO_S3:
        if not S3_BUCKET:
            logger.warning("UPLOAD_TO_S3=1 but S3_BUCKET is not set. Skipping S3 upload.")
        else:
            try:
                logger.info("Uploading run artifacts and per-run models to S3...")
                upload_dir_to_s3(run_dir, S3_BUCKET, f"{S3_PREFIX}/runs/{run_id}")
                upload_dir_to_s3(run_models_dir, S3_BUCKET, f"{S3_PREFIX}/models/{run_id}")

                s3 = boto3.client("s3")
                s3.upload_file(latest_json_path, S3_BUCKET, f"{S3_PREFIX}/latest_run.json")

                logger.info("S3 upload completed successfully.")
            except Exception as e:
                logger.exception(f"S3 upload failed (local artifacts still saved). Error: {e}")

    logger.info(f"Training completed successfully for run: {run_id}")


if __name__ == "__main__":
    main()
