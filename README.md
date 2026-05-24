# Fire Detection — Inference Server

A simple Flask web server that lets you upload a trained MobileNetV2 model (`.pt`) and run fire/non-fire inference on images through a browser UI.

This project is for students learning **cloud deployment on Azure**. One student trains the model in Google Colab, another deploys this server so the whole class can test it.

---

## How It Works

```
Student A (Google Colab)         This Server                     Your Browser
────────────────────────         ─────────────────────────────   ─────────────────
Train MobileNetV2            →   POST /upload_model  (best.pt)
Export best.pt                   POST /predict       (image)   ←  upload & test
                                 GET  /              (UI)      ←  http://<IP>
```

---

## Plan A — Deploy on Azure VM (Recommended)

Good for sharing with the whole class. Everyone just opens a URL.

### Step 1 — Create the VM

```bash
# Log in to Azure
az login

# Create a resource group
az group create --name fire-rg --location eastasia

# Create a VM (2 vCPU, 4 GB RAM — enough for CPU inference)
az vm create \
  --resource-group fire-rg \
  --name fire-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Open port 80
az vm open-port --resource-group fire-rg --name fire-vm --port 80

# Get the public IP
az vm show -d --resource-group fire-rg --name fire-vm --query publicIps -o tsv
```

### Step 2 — Install Docker on the VM

```bash
# SSH into the VM
ssh azureuser@<YOUR-VM-IP>

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3 — Clone and Start

```bash
git clone https://github.com/joe50304/azure_cloud_inference.git
cd azure_cloud_inference
docker compose up -d
```

First build takes ~5 minutes (downloading PyTorch CPU wheels). Once done, the server is live at `http://<YOUR-VM-IP>`.

### Step 4 — Test It

1. Open `http://<YOUR-VM-IP>` in your browser
2. Upload your `best.pt` from Colab
3. Upload a test image and hit **Start Inference**

### Cost Estimate (Azure Student $100 Credits)

| Resource | Spec | Est. Cost |
|----------|------|-----------|
| VM | Standard_B2s | ~$30/mo |
| Public IP | Standard | ~$4/mo |
| Disk | 30 GB OS | ~$2/mo |
| **Total** | | ~**$36/mo** |

> **Tip:** Run `az vm deallocate --resource-group fire-rg --name fire-vm` when you're done. You only pay for storage (~$2/mo) while it's stopped.

---

## Plan B — Run Locally (Windows / Mac)

Use this if you don't have Azure access or just want to test on your own machine.

### Option B-1 — Docker Desktop (Easiest)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Open a terminal and run:

```bash
git clone https://github.com/joe50304/azure_cloud_inference.git
cd azure_cloud_inference
docker compose up -d
```

3. Open `http://localhost` in your browser. Done.

### Option B-2 — Plain Python (No Docker)

Use this if Docker Desktop is not available.

```bash
git clone https://github.com/joe50304/azure_cloud_inference.git
cd azure_cloud_inference

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies (CPU-only PyTorch, ~500 MB)
pip install -r requirements.txt

# Run the server
python app.py
```

Open `http://localhost:5000` in your browser.

> **Note:** With this option there's no Nginx, so use port `5000` not `80`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/model_status` | Check if model is loaded |
| POST | `/upload_model` | Upload `best.pt` (form-data key: `model`) |
| POST | `/predict` | Run inference (form-data key: `image`) |

Example response from `/predict`:
```json
{
  "success": true,
  "prediction": "fire",
  "confidence": 97.43,
  "probabilities": { "fire": 97.43, "non-fire": 2.57 },
  "image_b64": "data:image/jpeg;base64,..."
}
```

---

## Project Structure

```
azure_cloud_inference/
├── app.py                      # Flask backend
├── templates/
│   └── index.html              # Frontend UI
├── requirements.txt            # Python dependencies (CPU-only torch)
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Flask + Nginx services
├── nginx.conf                  # Reverse proxy config
├── colab_training_template.py  # Template for the Colab student
└── .gitignore
```

---

## Useful Docker Commands

```bash
docker compose up -d            # Start in background
docker compose down             # Stop
docker compose ps               # Check status
docker compose logs -f          # Stream logs
docker compose up -d --build    # Rebuild after code changes
docker compose down -v          # Stop and delete model volume (resets uploaded model)
```
