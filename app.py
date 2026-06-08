import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
RAW_FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))
ALLOW_MODEL_UPLOAD = os.environ.get("ALLOW_MODEL_UPLOAD", "1") == "1"


def configured_path(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, default))
    return path if path.is_absolute() else BASE_DIR / path


MODEL_PATH = configured_path("MODEL_PATH", "models/primary_mlp.pt")
PREPROCESSING_PATH = configured_path(
    "PREPROCESSING_PATH", "models/preprocessing.joblib"
)
MODEL_MANIFEST_PATH = configured_path(
    "MODEL_MANIFEST_PATH", "models/model_manifest.json"
)
SAMPLES_PATH = configured_path("SAMPLES_PATH", "sample_transactions.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", str(20 * 1024 * 1024))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lock = threading.Lock()
model = None
preprocessing = None
model_metadata = {}
model_load_error = None


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


class InputValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preprocessing_hash():
    if os.environ.get("VERIFY_MODEL_HASHES", "1") == "0":
        return
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["artifact_sha256"]["preprocessing.joblib"]
    actual = sha256_file(PREPROCESSING_PATH)
    if actual.lower() != expected.lower():
        raise ValueError("SHA-256 verification failed for preprocessing.joblib")


def load_preprocessing_artifact():
    loaded_preprocessing = joblib.load(PREPROCESSING_PATH)
    required_keys = {"numeric_scaler", "log_scaler", "feature_columns"}
    if not required_keys.issubset(loaded_preprocessing):
        raise ValueError("Preprocessing artifact is missing required entries")
    return loaded_preprocessing


def build_model_from_checkpoint(path: Path, loaded_preprocessing):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required_keys = {
        "model_state_dict",
        "feature_columns",
        "threshold",
        "architecture",
    }
    if not required_keys.issubset(checkpoint):
        raise ValueError("Model checkpoint is missing required entries")
    if checkpoint["architecture"] != "FraudMLP(96-48-16)":
        raise ValueError("Only FraudMLP(96-48-16) checkpoints are accepted")
    if checkpoint["feature_columns"] != loaded_preprocessing["feature_columns"]:
        raise ValueError("Model and preprocessing feature columns do not match")
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Model threshold must be a finite number between 0 and 1")

    loaded_model = FraudMLP(input_dim=len(checkpoint["feature_columns"]))
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    if any(not torch.isfinite(parameter).all() for parameter in loaded_model.parameters()):
        raise ValueError("Model parameters must contain only finite values")
    loaded_model.to(device)
    loaded_model.eval()
    metadata = {
        "architecture": checkpoint["architecture"],
        "feature_columns": checkpoint["feature_columns"],
        "threshold": threshold,
        "artifact_sha256": sha256_file(path),
    }
    return loaded_model, metadata


def load_model_artifacts() -> bool:
    global model, preprocessing, model_metadata, model_load_error
    model = None
    preprocessing = None
    model_metadata = {}
    model_load_error = None

    try:
        verify_preprocessing_hash()
        loaded_preprocessing = load_preprocessing_artifact()
        loaded_model, loaded_metadata = build_model_from_checkpoint(
            MODEL_PATH, loaded_preprocessing
        )
        model = loaded_model
        preprocessing = loaded_preprocessing
        model_metadata = loaded_metadata
        return True
    except Exception as exc:
        model_load_error = str(exc)
        return False


def parse_transactions(payload) -> pd.DataFrame:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "transactions" in payload:
        rows = payload["transactions"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise InputValidationError(
            "Request body must be a transaction object, a list, or "
            "an object containing a transactions list"
        )

    if not isinstance(rows, list) or not rows:
        raise InputValidationError("At least one transaction is required")
    if len(rows) > MAX_BATCH_SIZE:
        raise InputValidationError(
            f"Batch size {len(rows)} exceeds the limit of {MAX_BATCH_SIZE}"
        )

    allowed = set(RAW_FEATURE_COLUMNS)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InputValidationError(f"Transaction {index} must be an object")
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in row]
        unknown = sorted(set(row) - allowed)
        if missing:
            raise InputValidationError(
                f"Transaction {index} is missing columns: {', '.join(missing)}"
            )
        if unknown:
            raise InputValidationError(
                f"Transaction {index} has unknown columns: {', '.join(unknown)}"
            )

    frame = pd.DataFrame(rows, columns=RAW_FEATURE_COLUMNS)
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("All transaction values must be numeric") from exc

    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise InputValidationError("Transaction values must be finite numbers")
    if (frame["Time"] < 0).any():
        raise InputValidationError("Time must be greater than or equal to zero")
    if (frame["Amount"] < 0).any():
        raise InputValidationError("Amount must be greater than or equal to zero")
    return frame


def transform_features(raw_frame: pd.DataFrame) -> pd.DataFrame:
    engineered = raw_frame.copy()
    hour = (engineered["Time"] % 86400) / 3600.0
    engineered["Hour"] = hour
    engineered["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    engineered["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    engineered["log_Amount"] = np.log1p(engineered["Amount"])

    scaled_numeric = preprocessing["numeric_scaler"].transform(
        engineered[["Time", "Amount"]]
    )
    scaled_log_amount = preprocessing["log_scaler"].transform(
        engineered[["log_Amount"]]
    )

    transformed = pd.DataFrame(index=engineered.index)
    for column in [f"V{i}" for i in range(1, 29)]:
        transformed[column] = engineered[column].astype(float)
    transformed["Hour"] = engineered["Hour"].astype(float)
    transformed["Hour_sin"] = engineered["Hour_sin"].astype(float)
    transformed["Hour_cos"] = engineered["Hour_cos"].astype(float)
    transformed["scaled_Time"] = scaled_numeric[:, 0]
    transformed["scaled_Amount"] = scaled_numeric[:, 1]
    transformed["scaled_log_Amount"] = scaled_log_amount[:, 0]
    return transformed[model_metadata["feature_columns"]]


def run_inference(raw_frame: pd.DataFrame) -> list[dict]:
    transformed = transform_features(raw_frame)
    inputs = torch.tensor(
        transformed.to_numpy(dtype=np.float32), dtype=torch.float32, device=device
    )
    with torch.no_grad(), model_lock:
        active_model = model
        threshold = model_metadata["threshold"]
        probabilities = torch.sigmoid(active_model(inputs)).cpu().numpy()

    return [
        {
            "index": index,
            "prediction": "fraud" if probability >= threshold else "normal",
            "fraud_probability": round(float(probability), 8),
            "fraud_probability_percent": round(float(probability) * 100, 4),
            "threshold": threshold,
            "requires_human_review": bool(probability >= threshold),
        }
        for index, probability in enumerate(probabilities)
    ]


def load_samples() -> list[dict]:
    try:
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
        return samples if isinstance(samples, list) else []
    except (OSError, json.JSONDecodeError):
        return []


load_model_artifacts()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/model_status")
def model_status():
    return jsonify(
        {
            "loaded": model is not None,
            "device": str(device),
            "architecture": model_metadata.get("architecture"),
            "threshold": model_metadata.get("threshold"),
            "artifact_sha256": model_metadata.get("artifact_sha256"),
            "raw_feature_count": len(RAW_FEATURE_COLUMNS),
            "model_feature_count": len(model_metadata.get("feature_columns", [])),
            "model_upload_enabled": ALLOW_MODEL_UPLOAD,
            "error": model_load_error,
        }
    )


@app.get("/schema")
def schema():
    return jsonify(
        {
            "input_columns": RAW_FEATURE_COLUMNS,
            "max_batch_size": MAX_BATCH_SIZE,
            "samples": load_samples(),
        }
    )


@app.post("/predict")
def predict():
    if model is None or preprocessing is None:
        return (
            jsonify(
                {
                    "success": False,
                    "message": model_load_error or "Model artifacts are not loaded",
                }
            ),
            503,
        )
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            raise InputValidationError("Request body must contain valid JSON")
        results = run_inference(parse_transactions(payload))
        return jsonify(
            {
                "success": True,
                "count": len(results),
                "decision_policy": "Transactions at or above the threshold require human review.",
                "predictions": results,
            }
        )
    except InputValidationError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@app.post("/upload_model")
def upload_model():
    global model, model_metadata, model_load_error
    if not ALLOW_MODEL_UPLOAD:
        return jsonify({"success": False, "message": "Model upload is disabled"}), 403
    if preprocessing is None:
        return (
            jsonify(
                {
                    "success": False,
                    "message": model_load_error or "Preprocessing artifact is not loaded",
                }
            ),
            503,
        )

    uploaded_file = request.files.get("model")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"success": False, "message": "A model file is required"}), 400
    if not uploaded_file.filename.lower().endswith(".pt"):
        return jsonify({"success": False, "message": "Model file must use the .pt extension"}), 400

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent, prefix=".candidate-", suffix=".pt", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            uploaded_file.save(temp_file)
        candidate_model, candidate_metadata = build_model_from_checkpoint(
            temp_path, preprocessing
        )
        with model_lock:
            os.replace(temp_path, MODEL_PATH)
            model = candidate_model
            model_metadata = candidate_metadata
            model_load_error = None
        return jsonify(
            {
                "success": True,
                "message": "Compatible model uploaded and loaded",
                "architecture": model_metadata["architecture"],
                "threshold": model_metadata["threshold"],
                "artifact_sha256": model_metadata["artifact_sha256"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "message": f"Incompatible model: {exc}"}), 400
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"success": False, "message": "Request body or model file is too large"}), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
