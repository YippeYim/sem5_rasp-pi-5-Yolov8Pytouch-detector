from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from flask_cors import CORS # <--- 1. นำเข้า CORS

# ต้องติดตั้ง: pip install flask-cors ultralytics numpy

app = Flask(__name__)
# อนุญาตให้ทุก Domain หรือ Origin เรียก API นี้ได้ (แก้ปัญหา CORS)
CORS(app) 

# 1. โหลดโมเดลเมื่อ Server เริ่มต้น (ครั้งเดียวเท่านั้น)
try:
    # ตรวจสอบให้แน่ใจว่าชื่อไฟล์โมเดลของคุณคือ 'best.pt'
    model = YOLO('best.pt') 
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

@app.route('/detect', methods=['POST']) # ใช้ Endpoint /detect
def detect_objects():
    if model is None:
        return jsonify({"error": "Model failed to load."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Please send a file named 'image'."}), 400

    file = request.files['image']
    
    # อ่านไฟล์ภาพและแปลงเป็น PIL Image
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format."}), 400

    # 4. รัน YOLO Inference (ตั้งค่า conf ตามความเหมาะสมของโมเดล)
    results = model(img, verbose=False, conf=0.5)

    # 5. แปลงผลลัพธ์จาก Ultralytics เป็น JSON Format
    detection_list = []
    
    for r in results:
        if r.boxes:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding Box coordinates (x1, y1, x2, y2)
            confs = r.boxes.conf.cpu().numpy()  # Confidence scores
            clss = r.boxes.cls.cpu().numpy()    # Class IDs

            for box, conf, cls_id in zip(boxes, confs, clss):
                detection_list.append({
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "label": model.names[int(cls_id)]
                })

    return jsonify({"detections": detection_list})

# ลบ if __name__ == '__main__':
# Server จะถูกรันด้วย PM2 + Gunicorn บนพอร์ต 1100 แทน
