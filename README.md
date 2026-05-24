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

## Plan A — Deploy on Azure VM

Good for sharing with the whole class — everyone just opens a URL.

> **No extra software needed.** Everything runs in your browser through the Azure Portal.

### Step 1 — Create a Virtual Machine

1. Go to [portal.azure.com](https://portal.azure.com) and sign in
2. Search for **Virtual machines** → click **Create → Azure virtual machine**
3. Fill in the basics:

   | Field | Value |
   |-------|-------|
   | Resource group | Create new → `fire-rg` |
   | Virtual machine name | `fire-vm` |
   | Region | East Asia (or wherever is closest) |
   | Image | **Ubuntu Server 22.04 LTS** |
   | Size | **Standard_B2s** (2 vCPU, 4 GB RAM) |
   | Authentication type | Password |
   | Username | `azureuser` |
   | Password | set your own |

4. Click **Next: Disks** → leave defaults
5. Click **Next: Networking** → make sure **Public IP** is set to `(new)`
6. Click **Review + create** → **Create**
7. Wait ~2 minutes for deployment to finish

### Step 2 — Open Port 80

1. Go to your new VM → click **Networking** in the left sidebar
2. Click **Add inbound port rule**
3. Set **Destination port ranges** to `80`, **Protocol** to `TCP`
4. Click **Add**

### Step 3 — Connect via Browser (No SSH Client Needed)

1. On your VM page, click **Connect → SSH using Azure CLI**
2. A terminal opens right in your browser — no PuTTY, no local SSH needed

   > If prompted, click **Configure** to enable the connection, then try again.

### Step 4 — Install Docker and Start the Server

Paste these commands into the browser terminal one by one:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Clone and start
git clone https://github.com/joe50304/azure_cloud_inference.git
cd azure_cloud_inference
docker compose up -d
```

First build takes about 5 minutes (downloading PyTorch). After that, check everything is running:

```bash
docker compose ps
```

Both `fire-inference` and `fire-nginx` should show `healthy` / `running`.

### Step 5 — Get Your Public IP

On the VM overview page in Azure Portal, copy the **Public IP address**. Share `http://<IP>` with the class.

### Step 6 — Test It

1. Open `http://<IP>` in your browser
2. Upload `best.pt` from Colab (Step 01 on the page)
3. Upload a test image → hit **Start Inference**

### Cost Estimate (Azure Student $100 Credits)

| Resource | Spec | Est. Cost |
|----------|------|-----------|
| VM | Standard_B2s | ~$30/mo |
| Public IP | Standard | ~$4/mo |
| Disk | 30 GB OS | ~$2/mo |
| **Total** | | ~**$36/mo** |

> **Save credits:** When you're done testing, go to your VM in the Portal → click **Stop**. You only pay for storage (~$2/mo) while it's stopped. Click **Start** to bring it back.

---

## Plan B — Run Locally (Windows / Mac)

Use this if you don't have Azure access, or just want to test on your own machine.

### Option B-1 — Docker Desktop (Easiest)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and make sure it's running
2. Open a terminal (PowerShell or Command Prompt on Windows) and run:

```bash
git clone https://github.com/joe50304/azure_cloud_inference.git
cd azure_cloud_inference
docker compose up -d
```

3. Open `http://localhost` in your browser. Done.

### Option B-2 — Plain Python (No Docker)

Use this if Docker Desktop isn't available.

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

Open `http://localhost:5000`. Done.

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
