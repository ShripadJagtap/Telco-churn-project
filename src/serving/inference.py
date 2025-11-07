import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# Try the Docker path first (used inside container)
# Fallbacks to local Mac path for development
MODEL_DIR = "/app/mlruns/0/local_manual_run/artifacts/model"

if not os.path.exists(MODEL_DIR):
    MODEL_DIR = "/Volumes/WD_BLACK SN7100 1TB/Documents/GitHub/Telco-churn-project/mlruns/0/local_manual_run/artifacts/model"

print(f"✅ Using model path: {MODEL_DIR}")

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from: {MODEL_DIR}")
except Exception as e:
    raise Exception(f"❌ Failed to load model from {MODEL_DIR}: {e}")


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that all columns are numeric before prediction."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Replace Yes/No etc. with numeric codes
    df = df.replace({"Yes": 1, "No": 0})

    # Convert all object/string columns to categorical codes
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # Fill NaNs that can appear from unseen categories
    df = df.fillna(0)

    return df


def predict(input_dict: dict) -> str:
    """Predict customer churn using the trained MLflow model."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    # enforce exact column order used when training
    expected_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    df_enc = df_enc.reindex(columns=expected_order, fill_value=0)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"
