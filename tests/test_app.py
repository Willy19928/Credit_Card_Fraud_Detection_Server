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


def test_predict_single_and_batch(client, samples):
    normal_response = client.post("/predict", json=samples[0]["transaction"])
    batch_response = client.post(
        "/predict",
        json={"transactions": [sample["transaction"] for sample in samples]},
    )

    assert normal_response.status_code == 200
    assert normal_response.get_json()["predictions"][0]["prediction"] == "normal"
    assert batch_response.status_code == 200
    assert batch_response.get_json()["count"] == 2


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
        inference_app.build_model_from_checkpoint(candidate_path, inference_app.preprocessing)
