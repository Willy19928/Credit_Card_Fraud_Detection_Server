import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import torch


RAW_FEATURE_COUNT = 30
EXPECTED_ARCHITECTURE = "FraudMLP(96-48-16)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def validate_artifacts(model_path: Path, preprocessing_path: Path) -> dict:
    preprocessing = joblib.load(preprocessing_path)
    required_preprocessing_keys = {"numeric_scaler", "log_scaler", "feature_columns"}
    if not required_preprocessing_keys.issubset(preprocessing):
        raise ValueError("preprocessing.joblib is missing required entries")

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

    threshold = float(checkpoint["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("checkpoint threshold must be finite and between 0 and 1")

    return {
        "architecture": checkpoint["architecture"],
        "threshold": threshold,
        "feature_count": len(checkpoint["feature_columns"]),
    }


def write_manifest(model_path: Path, preprocessing_path: Path, manifest_path: Path) -> None:
    metadata = validate_artifacts(model_path, preprocessing_path)
    manifest = {
        "model_name": "Primary NN - Tabular MLP",
        "architecture": metadata["architecture"],
        "decision_threshold": metadata["threshold"],
        "raw_feature_count": RAW_FEATURE_COUNT,
        "model_feature_count": metadata["feature_count"],
        "artifact_sha256": {
            "primary_mlp.pt": sha256_file(model_path),
            "preprocessing.joblib": sha256_file(preprocessing_path),
        },
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
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    preprocessing_path = resolve_path(args.preprocessing)
    manifest_path = resolve_path(args.manifest)
    write_manifest(model_path, preprocessing_path, manifest_path)
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
