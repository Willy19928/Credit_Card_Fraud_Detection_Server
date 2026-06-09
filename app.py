import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException


BASE_DIR = Path(__file__).resolve().parent
RAW_FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
MODEL_FEATURE_COLUMNS = [
    *[f"V{i}" for i in range(1, 29)],
    "Hour",
    "Hour_sin",
    "Hour_cos",
    "scaled_Time",
    "scaled_Amount",
    "scaled_log_Amount",
]
EXPECTED_ARCHITECTURE = "FraudMLP(96-48-16)"
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))
MAX_ABS_INPUT_VALUE = float(os.environ.get("MAX_ABS_INPUT_VALUE", "1000000000"))


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
RUN_METADATA_PATH = configured_path("RUN_METADATA_PATH", "models/run_metadata.json")
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


def load_model_manifest() -> dict:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    required_keys = {
        "architecture",
        "decision_threshold",
        "raw_feature_count",
        "model_feature_count",
        "artifact_sha256",
        "artifact_set_id",
    }
    if not required_keys.issubset(manifest):
        raise ValueError("Model manifest is missing required metadata")
    if manifest["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(f"Manifest architecture must be {EXPECTED_ARCHITECTURE}")
    if int(manifest["raw_feature_count"]) != len(RAW_FEATURE_COLUMNS):
        raise ValueError("Manifest raw feature count does not match server schema")
    if int(manifest["model_feature_count"]) != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("Manifest model feature count does not match server schema")
    threshold = float(manifest["decision_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Manifest decision threshold must be finite and between 0 and 1")
    return manifest


def verify_artifact_hashes(manifest: dict) -> None:
    if os.environ.get("VERIFY_MODEL_HASHES", "1") == "0":
        return
    expected_hashes = manifest.get("artifact_sha256", {})
    artifacts = {
        "primary_mlp.pt": MODEL_PATH,
        "preprocessing.joblib": PREPROCESSING_PATH,
    }
    if expected_hashes.get("run_metadata.json"):
        artifacts["run_metadata.json"] = RUN_METADATA_PATH
    for artifact_name, artifact_path in artifacts.items():
        expected = expected_hashes.get(artifact_name)
        if not expected:
            raise ValueError(f"Artifact manifest is missing SHA-256 for {artifact_name}")
        actual = sha256_file(artifact_path)
        if actual.lower() != expected.lower():
            raise ValueError(f"SHA-256 verification failed for {artifact_name}")


def validate_scaler(scaler, name: str, expected_feature_names: list[str]) -> None:
    if scaler is None or not hasattr(scaler, "transform"):
        raise ValueError(f"{name} must be a fitted scikit-learn transformer")
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError(f"{name} is not fitted")
    if int(scaler.n_features_in_) != len(expected_feature_names):
        raise ValueError(f"{name} expects an unexpected number of input features")
    if hasattr(scaler, "feature_names_in_"):
        actual_names = list(scaler.feature_names_in_)
        if actual_names != expected_feature_names:
            raise ValueError(f"{name} feature names do not match training schema")

    probe = pd.DataFrame(
        [[0.0] * len(expected_feature_names)], columns=expected_feature_names
    )
    transformed = np.asarray(scaler.transform(probe), dtype=np.float64)
    if transformed.shape != (1, len(expected_feature_names)):
        raise ValueError(f"{name} returned an unexpected transform shape")
    if not np.isfinite(transformed).all():
        raise ValueError(f"{name} produced non-finite values during validation")


def validate_preprocessing_artifact(loaded_preprocessing: dict) -> None:
    if not isinstance(loaded_preprocessing, dict):
        raise ValueError("Preprocessing artifact must be a dictionary")
    required_keys = {"numeric_scaler", "log_scaler", "feature_columns"}
    if not required_keys.issubset(loaded_preprocessing):
        raise ValueError("Preprocessing artifact is missing required entries")
    if list(loaded_preprocessing["feature_columns"]) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Preprocessing feature columns do not match server schema")
    validate_scaler(
        loaded_preprocessing["numeric_scaler"], "numeric_scaler", ["Time", "Amount"]
    )
    validate_scaler(loaded_preprocessing["log_scaler"], "log_scaler", ["log_Amount"])


def load_preprocessing_artifact():
    loaded_preprocessing = joblib.load(PREPROCESSING_PATH)
    validate_preprocessing_artifact(loaded_preprocessing)
    return loaded_preprocessing


def build_model_from_checkpoint(path: Path, loaded_preprocessing, manifest: dict):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required_keys = {
        "model_state_dict",
        "feature_columns",
        "threshold",
        "architecture",
    }
    if not required_keys.issubset(checkpoint):
        raise ValueError("Model checkpoint is missing required entries")
    if checkpoint["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(f"Only {EXPECTED_ARCHITECTURE} checkpoints are accepted")
    if checkpoint["feature_columns"] != loaded_preprocessing["feature_columns"]:
        raise ValueError("Model and preprocessing feature columns do not match")
    if checkpoint["feature_columns"] != MODEL_FEATURE_COLUMNS:
        raise ValueError("Checkpoint feature columns do not match server schema")
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Model threshold must be a finite number between 0 and 1")
    if not np.isclose(threshold, float(manifest["decision_threshold"]), atol=1e-12):
        raise ValueError("Manifest threshold does not match checkpoint threshold")
    if manifest["architecture"] != checkpoint["architecture"]:
        raise ValueError("Manifest architecture does not match checkpoint")
    if int(manifest["model_feature_count"]) != len(checkpoint["feature_columns"]):
        raise ValueError("Manifest model feature count does not match checkpoint")

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
        "artifact_sha256": manifest.get("artifact_sha256", {}),
        "artifact_set_id": manifest.get("artifact_set_id"),
        "training_run": manifest.get("training_run", {}),
    }
    return loaded_model, metadata


def load_model_artifacts() -> bool:
    global model, preprocessing, model_metadata, model_load_error
    model = None
    preprocessing = None
    model_metadata = {}
    model_load_error = None

    try:
        loaded_manifest = load_model_manifest()
        verify_artifact_hashes(loaded_manifest)
        loaded_preprocessing = load_preprocessing_artifact()
        loaded_model, loaded_metadata = build_model_from_checkpoint(
            MODEL_PATH, loaded_preprocessing, loaded_manifest
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
        for column in RAW_FEATURE_COLUMNS:
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise InputValidationError(
                    f"Transaction {index} column {column} must be a JSON number"
                )

    frame = pd.DataFrame(rows, columns=RAW_FEATURE_COLUMNS)
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("All transaction values must be numeric") from exc

    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise InputValidationError("Transaction values must be finite numbers")
    if (np.abs(frame.to_numpy(dtype=np.float64)) > MAX_ABS_INPUT_VALUE).any():
        raise InputValidationError(
            f"Transaction values must be within +/-{MAX_ABS_INPUT_VALUE:g}"
        )
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
    transformed = transformed[model_metadata["feature_columns"]]
    transformed_values = transformed.to_numpy(dtype=np.float64)
    if not np.isfinite(transformed_values).all():
        raise InputValidationError("Transformed feature values must be finite numbers")
    with np.errstate(over="ignore"):
        float32_values = transformed_values.astype(np.float32)
    if not np.isfinite(float32_values).all():
        raise InputValidationError("Transaction values are outside the supported model range")
    return transformed


def run_inference(raw_frame: pd.DataFrame) -> list[dict]:
    transformed = transform_features(raw_frame)
    inputs = torch.tensor(
        transformed.to_numpy(dtype=np.float32), dtype=torch.float32, device=device
    )
    with torch.no_grad(), model_lock:
        active_model = model
        threshold = model_metadata["threshold"]
        probabilities = torch.sigmoid(active_model(inputs)).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model produced a non-finite probability")

    return [
        {
            "index": index,
            "prediction": "fraud" if probability >= threshold else "normal",
            "fraud_probability": float(probability),
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


if not load_model_artifacts() and os.environ.get("FAIL_ON_MODEL_LOAD_ERROR", "1") != "0":
    raise RuntimeError("Model artifacts failed validation")


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
            "artifact_set_id": model_metadata.get("artifact_set_id"),
            "training_run": model_metadata.get("training_run"),
            "raw_feature_count": len(RAW_FEATURE_COLUMNS),
            "model_feature_count": len(model_metadata.get("feature_columns", [])),
            "model_update_mode": "replace artifacts before service startup",
            "error": "Model artifacts failed validation" if model_load_error else None,
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
                    "message": "Model artifacts are not loaded",
                }
            ),
            503,
        )
    try:
        request_id = uuid.uuid4().hex
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        payload = request.get_json(silent=True)
        if payload is None:
            raise InputValidationError("Request body must contain valid JSON")
        results = run_inference(parse_transactions(payload))
        return jsonify(
            {
                "success": True,
                "request_id": request_id,
                "timestamp_utc": timestamp_utc,
                "count": len(results),
                "model_version": model_metadata.get("artifact_set_id"),
                "model_artifact_sha256": model_metadata.get("artifact_sha256", {}).get(
                    "primary_mlp.pt"
                ),
                "decision_policy": "Transactions at or above the threshold require human review.",
                "predictions": results,
            }
        )
    except InputValidationError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"success": False, "message": "Request body is too large"}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return jsonify({"success": False, "message": error.description}), error.code
    app.logger.exception("Unhandled inference service error")
    return jsonify({"success": False, "message": "Inference service error"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
