import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


RAW_FEATURE_COUNT = 30
EXPECTED_ARCHITECTURE = "FraudMLP(96-48-16)"
EXPECTED_FEATURE_COLUMNS = [
    *[f"V{i}" for i in range(1, 29)],
    "Hour",
    "Hour_sin",
    "Hour_cos",
    "scaled_Time",
    "scaled_Amount",
    "scaled_log_Amount",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def validate_scaler(scaler, name: str, expected_feature_names: list[str]) -> None:
    if scaler is None or not hasattr(scaler, "transform"):
        raise ValueError(f"{name} must be a fitted scikit-learn transformer")
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError(f"{name} is not fitted")
    if int(scaler.n_features_in_) != len(expected_feature_names):
        raise ValueError(f"{name} has an unexpected input width")
    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != expected_feature_names:
            raise ValueError(f"{name} feature names do not match training schema")

    probe = pd.DataFrame(
        [[0.0] * len(expected_feature_names)], columns=expected_feature_names
    )
    transformed = np.asarray(scaler.transform(probe), dtype=np.float64)
    if transformed.shape != (1, len(expected_feature_names)):
        raise ValueError(f"{name} returned an unexpected transform shape")
    if not np.isfinite(transformed).all():
        raise ValueError(f"{name} produced non-finite values")


def read_run_metadata(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    dataset = metadata.get("dataset", {})
    primary_model = metadata.get("primary_model", {})
    return (
        {
            "run_id": metadata.get("run_id"),
            "artifact_set_id": metadata.get("artifact_set_id"),
            "run_timestamp_utc": metadata.get("run_timestamp_utc"),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "dataset_shape": dataset.get("shape"),
            "duplicate_rows": dataset.get("duplicate_rows"),
            "split_strategy": "stratified random train/validation/test split",
            "primary_model": {
                "name": primary_model.get("name"),
                "architecture": primary_model.get("architecture"),
                "threshold": primary_model.get("threshold"),
            },
        },
        sha256_file(path),
    )


def validate_artifacts(
    model_path: Path, preprocessing_path: Path, run_metadata_path: Path
) -> dict:
    preprocessing = joblib.load(preprocessing_path)
    required_preprocessing_keys = {"numeric_scaler", "log_scaler", "feature_columns"}
    if not required_preprocessing_keys.issubset(preprocessing):
        raise ValueError("preprocessing.joblib is missing required entries")
    if list(preprocessing["feature_columns"]) != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("preprocessing feature columns do not match server schema")
    validate_scaler(preprocessing["numeric_scaler"], "numeric_scaler", ["Time", "Amount"])
    validate_scaler(preprocessing["log_scaler"], "log_scaler", ["log_Amount"])

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    required_checkpoint_keys = {
        "model_state_dict",
        "feature_columns",
        "threshold",
        "architecture",
    }
    if not required_checkpoint_keys.issubset(checkpoint):
        raise ValueError("primary_mlp.pt is missing required checkpoint entries")
    if checkpoint["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(f"checkpoint architecture must be {EXPECTED_ARCHITECTURE}")
    if checkpoint["feature_columns"] != preprocessing["feature_columns"]:
        raise ValueError("checkpoint and preprocessing feature columns do not match")
    if checkpoint["feature_columns"] != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("checkpoint feature columns do not match server schema")

    threshold = float(checkpoint["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("checkpoint threshold must be finite and between 0 and 1")
    if any(
        not torch.isfinite(tensor).all()
        for tensor in checkpoint["model_state_dict"].values()
        if torch.is_tensor(tensor)
    ):
        raise ValueError("checkpoint model parameters must be finite")

    training_run, run_metadata_sha256 = read_run_metadata(run_metadata_path)

    return {
        "architecture": checkpoint["architecture"],
        "threshold": threshold,
        "feature_count": len(checkpoint["feature_columns"]),
        "training_run": training_run,
        "run_metadata_sha256": run_metadata_sha256,
    }


def write_manifest(
    model_path: Path,
    preprocessing_path: Path,
    manifest_path: Path,
    run_metadata_path: Path,
) -> None:
    metadata = validate_artifacts(model_path, preprocessing_path, run_metadata_path)
    model_hash = sha256_file(model_path)
    preprocessing_hash = sha256_file(preprocessing_path)
    artifact_hashes = {
        "primary_mlp.pt": model_hash,
        "preprocessing.joblib": preprocessing_hash,
    }
    if metadata["run_metadata_sha256"]:
        artifact_hashes["run_metadata.json"] = metadata["run_metadata_sha256"]
    artifact_set_id = metadata["training_run"].get("artifact_set_id") or (
        "sha256:" + sha256_text("|".join(sorted(artifact_hashes.values())))
    )
    manifest = {
        "model_name": "Primary NN - Tabular MLP",
        "architecture": metadata["architecture"],
        "decision_threshold": metadata["threshold"],
        "raw_feature_count": RAW_FEATURE_COUNT,
        "model_feature_count": metadata["feature_count"],
        "artifact_sha256": artifact_hashes,
        "artifact_set_id": artifact_set_id,
        "training_run": metadata["training_run"],
        "source_project": "AISec Final - SDG 8 Credit Card Fraud Detection",
        "intended_use": "Classroom fraud-risk scoring with human review",
        "update_mode": "offline artifact replacement before service startup",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate offline model artifacts and update models/model_manifest.json."
    )
    parser.add_argument("--model", default="models/primary_mlp.pt")
    parser.add_argument("--preprocessing", default="models/preprocessing.joblib")
    parser.add_argument("--manifest", default="models/model_manifest.json")
    parser.add_argument(
        "--run-metadata",
        default="models/run_metadata.json",
        help="Optional run metadata copied from the training repository.",
    )
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    preprocessing_path = resolve_path(args.preprocessing)
    manifest_path = resolve_path(args.manifest)
    run_metadata_path = resolve_path(args.run_metadata)
    write_manifest(model_path, preprocessing_path, manifest_path, run_metadata_path)
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
