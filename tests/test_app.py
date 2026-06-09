import io
import json

import pytest

import app as inference_app


@pytest.fixture()
def client():
    inference_app.app.config.update(TESTING=True)
    return inference_app.app.test_client()


@pytest.fixture()
def samples():
    return json.loads(inference_app.SAMPLES_PATH.read_text(encoding="utf-8"))


def test_model_status_reports_loaded_model(client):
    response = client.get("/model_status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["loaded"] is True
    assert payload["architecture"] == "FraudMLP(96-48-16)"
    assert payload["model_feature_count"] == 34
    assert payload["model_update_mode"] == "replace artifacts before service startup"
    assert payload["artifact_set_id"].startswith("sha256:")
    assert payload["training_run"]["dataset_sha256"]


def test_predict_single_and_batch(client, samples):
    normal_response = client.post("/predict", json=samples[0]["transaction"])
    batch_response = client.post(
        "/predict",
        json={"transactions": [sample["transaction"] for sample in samples]},
    )

    assert normal_response.status_code == 200
    normal_payload = normal_response.get_json()
    assert normal_payload["predictions"][0]["prediction"] == "normal"
    assert normal_payload["request_id"]
    assert normal_payload["model_version"].startswith("sha256:")
    assert batch_response.status_code == 200
    assert batch_response.get_json()["count"] == 2


def test_predict_returns_unrounded_probability_for_threshold_decision(client, samples):
    response = client.post("/predict", json=samples[1]["transaction"])
    prediction = response.get_json()["predictions"][0]

    assert response.status_code == 200
    if prediction["prediction"] == "fraud":
        assert prediction["fraud_probability"] >= prediction["threshold"]


def test_predict_rejects_missing_and_unknown_columns(client, samples):
    missing = dict(samples[0]["transaction"])
    missing.pop("V28")
    unknown = dict(samples[0]["transaction"], Class=0)

    missing_response = client.post("/predict", json=missing)
    unknown_response = client.post("/predict", json=unknown)

    assert missing_response.status_code == 400
    assert "missing columns: V28" in missing_response.get_json()["message"]
    assert unknown_response.status_code == 400
    assert "unknown columns: Class" in unknown_response.get_json()["message"]


def test_predict_rejects_strings_booleans_and_extreme_values(client, samples):
    numeric_string = dict(samples[0]["transaction"])
    numeric_string["Amount"] = "149.62"
    boolean_value = dict(samples[0]["transaction"])
    boolean_value["V1"] = True
    extreme_value = dict(samples[0]["transaction"])
    extreme_value["V1"] = 1e308

    string_response = client.post("/predict", json=numeric_string)
    boolean_response = client.post("/predict", json=boolean_value)
    extreme_response = client.post("/predict", json=extreme_value)

    assert string_response.status_code == 400
    assert "must be a JSON number" in string_response.get_json()["message"]
    assert boolean_response.status_code == 400
    assert "must be a JSON number" in boolean_response.get_json()["message"]
    assert extreme_response.status_code == 400
    assert "within +/-" in extreme_response.get_json()["message"]


def test_upload_endpoint_is_not_available(client):
    response = client.post(
        "/upload_model",
        data={"model": (io.BytesIO(b"not a checkpoint"), "model.pt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 404


def test_checkpoint_validation_rejects_invalid_threshold(tmp_path):
    checkpoint = inference_app.torch.load(
        inference_app.MODEL_PATH, map_location="cpu", weights_only=True
    )
    checkpoint["threshold"] = 2.0
    candidate_path = tmp_path / "invalid-threshold.pt"
    inference_app.torch.save(checkpoint, candidate_path)

    with pytest.raises(ValueError, match="between 0 and 1"):
        inference_app.build_model_from_checkpoint(
            candidate_path,
            inference_app.preprocessing,
            inference_app.load_model_manifest(),
        )


def test_text_artifact_hash_is_stable_across_line_endings(tmp_path):
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(b'{\n  "value": 1\n}\n')
    crlf_path.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')

    assert inference_app.sha256_artifact(lf_path) == inference_app.sha256_artifact(
        crlf_path
    )
    assert inference_app.sha256_file(lf_path) != inference_app.sha256_file(crlf_path)


def test_preprocessing_validation_rejects_missing_scaler():
    bad_preprocessing = {
        "numeric_scaler": None,
        "log_scaler": inference_app.preprocessing["log_scaler"],
        "feature_columns": inference_app.MODEL_FEATURE_COLUMNS,
    }

    with pytest.raises(ValueError, match="numeric_scaler"):
        inference_app.validate_preprocessing_artifact(bad_preprocessing)


def test_manifest_validation_rejects_metadata_mismatch(tmp_path, monkeypatch):
    manifest = inference_app.load_model_manifest()
    broken_manifest = dict(manifest, model_feature_count=999)
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps(broken_manifest), encoding="utf-8")
    monkeypatch.setattr(inference_app, "MODEL_MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="model feature count"):
        inference_app.load_model_manifest()


def test_unexpected_inference_error_returns_json_without_internal_details(
    client, samples, monkeypatch
):
    def fail_inference(_frame):
        raise RuntimeError("secret path C:\\internal\\model.joblib")

    monkeypatch.setattr(inference_app, "run_inference", fail_inference)
    response = client.post("/predict", json=samples[0]["transaction"])
    payload = response.get_json()

    assert response.status_code == 500
    assert response.content_type.startswith("application/json")
    assert payload == {"message": "Inference service error", "success": False}


def test_model_status_hides_internal_load_error(client, monkeypatch):
    monkeypatch.setattr(inference_app, "model", None)
    monkeypatch.setattr(inference_app, "model_load_error", "C:\\secret\\artifact.pt")

    response = client.get("/model_status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["loaded"] is False
    assert payload["error"] == "Model artifacts failed validation"


def test_predict_hides_internal_load_error(client, samples, monkeypatch):
    monkeypatch.setattr(inference_app, "model", None)
    monkeypatch.setattr(inference_app, "preprocessing", None)
    monkeypatch.setattr(inference_app, "model_load_error", "C:\\secret\\artifact.pt")

    response = client.post("/predict", json=samples[0]["transaction"])
    payload = response.get_json()

    assert response.status_code == 503
    assert payload == {"message": "Model artifacts are not loaded", "success": False}
