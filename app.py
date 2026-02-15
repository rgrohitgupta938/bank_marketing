# app.py

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from src.s3_artifacts import (
    read_latest_run_id,
    load_preprocessor,
    load_model,
    clear_cache,
)

MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def _to_dense_if_needed(X, force_dense: bool):
    """
    Naive Bayes expects dense arrays. Others can handle sparse.
    """
    if not force_dense:
        return X
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


@st.cache_resource(show_spinner=False)
def _load_artifacts_cached(bucket: str, prefix: str, run_id: str, model_filename: str):
    pre = load_preprocessor(bucket, prefix, run_id)
    model = load_model(bucket, prefix, run_id, model_filename)
    return pre, model


def main():
    st.set_page_config(page_title="Bank Marketing - Model Comparator (S3)", layout="wide")

    st.title("Bank Marketing — Term Deposit Subscription Prediction (S3)")
    st.write(
        "Upload a CSV file with Bank Marketing features and choose a trained model to generate predictions. "
        "If your uploaded CSV also contains column **y** (yes/no or 0/1), the app will compute evaluation metrics.\n\n"
        "This app loads the **latest** model artifacts from **AWS S3** using `latest_run.json`."
    )

    # -----------------------------
    # Sidebar: S3 + model controls
    # -----------------------------
    st.sidebar.header("S3 Settings")

    bucket = st.secrets.get("S3_BUCKET", os.getenv("S3_BUCKET", ""))
    prefix = st.secrets.get("S3_PREFIX", os.getenv("S3_PREFIX", "bank-marketing"))

    if not bucket:
        st.error("S3_BUCKET is not set. Add it in Streamlit secrets or environment variables.")
        st.stop()

    st.sidebar.caption(f"Bucket: {bucket}")
    st.sidebar.caption(f"Prefix: {prefix}")

    use_latest = st.sidebar.checkbox("Use latest_run.json", value=True)

    if use_latest:
        try:
            run_id = read_latest_run_id(bucket, prefix)
            st.sidebar.success(f"Latest run_id: {run_id}")
        except Exception as e:
            st.sidebar.error("Failed to read latest_run.json from S3.")
            st.sidebar.exception(e)
            st.stop()
    else:
        run_id = st.sidebar.text_input("Run ID (manual)", value="")
        if not run_id.strip():
            st.sidebar.warning("Enter a run_id.")
            st.stop()

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
    st.sidebar.caption("Tip: Logistic often gives best recall; XGBoost often gives best AUC.")

    if st.sidebar.button("Clear cache"):
        clear_cache()
        _load_artifacts_cached.clear()
        st.sidebar.success("Cache cleared.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return

    # Load input
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Handle target if present
    y_true = None
    if "y" in df.columns:
        y_col = df["y"].copy()
        if y_col.dtype == object:
            y_col = y_col.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        y_true = pd.to_numeric(y_col, errors="coerce")
        df = df.drop(columns=["y"])

    # Load artifacts from S3 (cached)
    model_filename = MODEL_FILES[model_choice]
    with st.spinner("Loading model artifacts from S3..."):
        try:
            preprocessor, model = _load_artifacts_cached(bucket, prefix, run_id, model_filename)
        except Exception as e:
            st.error("Failed to load preprocessor/model from S3.")
            st.exception(e)
            return

    # Transform features
    try:
        X_proc = preprocessor.transform(df)
    except Exception as e:
        st.error(
            "Preprocessing failed. This usually means your uploaded CSV columns don't match training schema."
        )
        st.exception(e)
        return

    # Predict
    force_dense = (model_choice == "Naive Bayes")
    X_infer = _to_dense_if_needed(X_proc, force_dense=force_dense)

    try:
        y_prob = model.predict_proba(X_infer)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        return

    # Output predictions
    out = df.copy()
    out["pred_prob"] = y_prob
    out["pred_label"] = y_pred

    st.subheader("Predictions")
    st.caption(f"Run ID: {run_id} | Model: {model_choice}")
    st.dataframe(out.head(30), use_container_width=True)

    # Download
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

    # If ground truth exists, show metrics + confusion matrix + report
    if y_true is not None and y_true.notna().all():
        st.subheader("Evaluation (because your upload included y)")
        metrics = _compute_metrics(y_true.values, y_pred, y_prob)

        c1, c2, c3 = st.columns(3)
        keys = list(metrics.keys())
        for i, k in enumerate(keys):
            val = metrics[k]
            if i % 3 == 0:
                c1.metric(k, f"{val:.4f}")
            elif i % 3 == 1:
                c2.metric(k, f"{val:.4f}")
            else:
                c3.metric(k, f"{val:.4f}")

        cm = confusion_matrix(y_true.values, y_pred)
        st.write("Confusion Matrix (rows=actual, cols=predicted)")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        st.write("Classification Report")
        report = classification_report(y_true.values, y_pred, digits=4)
        st.code(report)
    else:
        st.warning(
            "No valid `y` column found in upload (or it contains missing/invalid values), "
            "so evaluation metrics are not computed."
        )

    st.caption("Models trained on bank-full.csv. Target: y=1 means client subscribes to term deposit.")


if __name__ == "__main__":
    main()
