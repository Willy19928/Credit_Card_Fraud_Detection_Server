from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# 全域設定
# ─────────────────────────────────────────
CLASS_NAMES = ['fire', 'non-fire']
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.pt')  # Docker volume: /app/models/best.pt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None  # 延遲載入

DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    """載入模型（若 best.pt 存在）"""
    global model
    if not os.path.exists(MODEL_PATH):
        return False
    m = models.mobilenet_v2(weights=None)
    num_ftrs = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m.to(device)
    m.eval()
    model = m
    return True

# 啟動時嘗試載入
load_model()

# ─────────────────────────────────────────
# 路由
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """學生上傳 .pt 模型檔案"""
    if 'model' not in request.files:
        return jsonify({'success': False, 'message': '未找到模型檔案'}), 400
    
    file = request.files['model']
    if not file.filename.endswith('.pt'):
        return jsonify({'success': False, 'message': '請上傳 .pt 格式的模型'}), 400
    
    file.save(MODEL_PATH)
    
    if load_model():
        return jsonify({'success': True, 'message': '模型載入成功！'})
    else:
        return jsonify({'success': False, 'message': '模型載入失敗，請確認格式正確'}), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """檢查模型是否已載入"""
    return jsonify({
        'loaded': model is not None,
        'device': str(device),
        'classes': CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """推論端點：接受圖片，回傳預測結果"""
    if model is None:
        return jsonify({'success': False, 'message': '模型尚未載入，請先上傳 best.pt'}), 400
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '未找到圖片'}), 400
    
    file = request.files['image']
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = DATA_TRANSFORMS(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        result = CLASS_NAMES[preds[0]]
        all_probs = {CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2) 
                     for i in range(len(CLASS_NAMES))}
        confidence = round(probabilities[preds[0]].item() * 100, 2)
        
        # 回傳圖片預覽（base64）
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        ext = file.content_type.split('/')[-1]
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': confidence,
            'probabilities': all_probs,
            'image_b64': f'data:{file.content_type};base64,{img_b64}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'推論錯誤：{str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
