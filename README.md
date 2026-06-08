# Credit Card Fraud Detection - Azure Inference Server

This project was forked from
[joe50304/azure_cloud_inference](https://github.com/joe50304/azure_cloud_inference)
and modified for the AISec Final credit-card fraud detection project. The
original MobileNet image-classification inference workflow was replaced with a
Flask-based tabular transaction scoring service.

The service:

- loads the packaged `FraudMLP(96-48-16)` checkpoint and preprocessing artifact;
- accepts one transaction or a JSON batch;
- reproduces the notebook's feature engineering and scaling;
- uses the checkpoint's validation-selected threshold;
- routes flagged transactions to human review;
- lets users upload a compatible `.pt` checkpoint through the web UI or API.

## Model Contract

Raw requests must contain exactly these 30 numeric fields:

```text
Time, V1, V2, ... V28, Amount
```

The server uses the packaged `preprocessing.joblib` to create the model's 34
features:

```text
V1 ... V28, Hour, Hour_sin, Hour_cos,
scaled_Time, scaled_Amount, scaled_log_Amount
```

The deployed model artifact is accepted only when it contains:

- architecture: `FraudMLP(96-48-16)`;
- `model_state_dict`;
- the same 34 `feature_columns` as the packaged preprocessing artifact;
- a decision `threshold`.

Uploaded checkpoints are loaded with `torch.load(..., weights_only=True)`.
Uploading a replacement preprocessing artifact is intentionally unsupported.

## Included Artifacts

| File | Purpose |
| --- | --- |
| `models/primary_mlp.pt` | Primary tabular neural-network checkpoint |
| `models/preprocessing.joblib` | Training-fitted scalers and feature order |
| `models/model_manifest.json` | Artifact metadata and preprocessing SHA-256 |
| `sample_transactions.json` | Public-dataset examples used by the UI |

Packaged model threshold: `0.9838983416557312`.

## Run Locally

Python 3.10 or later is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Run With Docker

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Open `http://localhost`.

The Docker named volume `model_storage` preserves user-uploaded compatible
models across container restarts. On startup, the container copies only missing
default artifacts into the volume and does not overwrite an uploaded
`primary_mlp.pt`.

Reset to the checkpoint packaged in the image:

```bash
docker compose down -v
docker compose up -d --build
```

## Deploy On An Azure VM

Create an Ubuntu VM, allow inbound TCP port `80`, then run:

```bash
sudo apt update
sudo apt install -y git
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

git clone https://github.com/Willy19928/Credit_Card_Fraud_Detection_Server.git
cd Credit_Card_Fraud_Detection_Server
docker compose up -d --build
```

Open `http://<VM_PUBLIC_IP>`.

Stop or deallocate the VM when it is not in use to avoid unnecessary charges.

## API

### Status

```http
GET /model_status
```

Returns the loaded architecture, threshold, device, artifact SHA-256, and upload
availability.

### Schema And Samples

```http
GET /schema
```

Returns required input columns, batch limit, and UI sample transactions.

### Single Prediction

```http
POST /predict
Content-Type: application/json
```

```json
{
  "Time": 0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62
}
```

Example response:

```json
{
  "success": true,
  "count": 1,
  "predictions": [
    {
      "index": 0,
      "prediction": "normal",
      "fraud_probability": 0.05713778,
      "fraud_probability_percent": 5.7138,
      "threshold": 0.9838983416557312,
      "requires_human_review": false
    }
  ]
}
```

### Batch Prediction

Send a JSON list or wrap it in a `transactions` object:

```json
{
  "transactions": [
    { "Time": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0, "Amount": 0 }
  ]
}
```

Default batch limit: `1000`.

### Upload A Compatible Model

```http
POST /upload_model
Content-Type: multipart/form-data
form field: model
```

```bash
curl -F "model=@models/primary_mlp.pt" http://localhost:5000/upload_model
```

The service validates the checkpoint before atomically replacing the active
model. Docker uses one Gunicorn worker so the newly uploaded in-memory model is
used consistently by following requests.

## Configuration

| Environment variable | Default |
| --- | --- |
| `MODEL_PATH` | `models/primary_mlp.pt` |
| `PREPROCESSING_PATH` | `models/preprocessing.joblib` |
| `MODEL_MANIFEST_PATH` | `models/model_manifest.json` |
| `ALLOW_MODEL_UPLOAD` | `1` |
| `VERIFY_MODEL_HASHES` | `1` |
| `MAX_BATCH_SIZE` | `1000` |
| `MAX_CONTENT_LENGTH` | `20971520` bytes |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The classifier is intended for classroom risk scoring with human review. It
must not automatically block accounts or punish customers without production
governance, monitoring, and incident-response controls.
