# 🔥 Fire Detection — Google Colab 訓練筆記本
# 複製此 notebook 到自己的 Colab 並執行

# ──────────────────────────────────────────────────
# CELL 1: 安裝套件
# ──────────────────────────────────────────────────
# !pip install torch torchvision

# ──────────────────────────────────────────────────
# CELL 2: 掛載 Google Drive（存放資料集）
# ──────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ──────────────────────────────────────────────────
# CELL 3: 訓練程式碼
# ──────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os, copy, time

# ── 資料集路徑（請修改成你的路徑）──
# 資料夾結構：
# /content/drive/MyDrive/fire_dataset/
#     train/
#         fire/        ← 火災圖片
#         non-fire/    ← 非火災圖片
#     val/
#         fire/
#         non-fire/

DATA_DIR = '/content/drive/MyDrive/fire_dataset'  # ← 修改這裡
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

# ── 資料增強 ──
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders    = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                  for x in ['train', 'val']}
class_names    = image_datasets['train'].classes
print(f"類別：{class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ── 模型：MobileNetV2 遷移學習 ──
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():      # 凍結 backbone
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ── 訓練迴圈 ──
best_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} {'─'*30}")
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        running_loss, running_correct = 0.0, 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss    += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

        if phase == 'train': scheduler.step()
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc  = running_correct.double() / len(image_datasets[phase])
        print(f"  {phase:5s} | loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, 'best.pt')
            print(f"  ✓ 儲存最佳模型（acc={best_acc:.4f}）")

print(f"\n訓練完成！最佳驗證準確率：{best_acc:.4f}")
print("模型已儲存為 best.pt")

# ──────────────────────────────────────────────────
# CELL 4: 下載 best.pt 到本機
# ──────────────────────────────────────────────────
# from google.colab import files
# files.download('best.pt')
