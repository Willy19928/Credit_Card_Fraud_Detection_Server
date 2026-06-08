import io
import json
import shutil

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
    assert payload["model_upload_enabled"] is True


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


def test_upload_rejects_non_pt_file(client):
    response = client.post(
        "/upload_model",
        data={"model": (io.BytesIO(b"not a checkpoint"), "model.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert ".pt extension" in response.get_json()["message"]


def test_upload_accepts_compatible_checkpoint(client, tmp_path, monkeypatch):
    original_model_path = inference_app.MODEL_PATH
    temporary_model_path = tmp_path / "primary_mlp.pt"
    shutil.copy2(original_model_path, temporary_model_path)
    monkeypatch.setattr(inference_app, "MODEL_PATH", temporary_model_path)

    response = client.post(
        "/upload_model",
        data={
            "model": (
                io.BytesIO(original_model_path.read_bytes()),
                "replacement.pt",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert response.get_json()["architecture"] == "FraudMLP(96-48-16)"


def test_upload_rejects_invalid_threshold(client):
    checkpoint = inference_app.torch.load(
        inference_app.MODEL_PATH, map_location="cpu", weights_only=True
    )
    checkpoint["threshold"] = 2.0
    buffer = io.BytesIO()
    inference_app.torch.save(checkpoint, buffer)
    buffer.seek(0)

    response = client.post(
        "/upload_model",
        data={"model": (buffer, "invalid-threshold.pt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "between 0 and 1" in response.get_json()["message"]
