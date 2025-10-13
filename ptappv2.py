from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from flask_cors import CORS 
import math 

# ต้องติดตั้ง: pip install flask-cors ultralytics numpy

app = Flask(__name__)
CORS(app) 

# 1. โหลดโมเดลเมื่อ Server เริ่มต้น (ครั้งเดียวเท่านั้น)
try:
    model = YOLO('bestv2.pt') 
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# --- ฟังก์ชันช่วยในการคำนวณ ---
def get_center_point(box):
    """คำนวณจุดศูนย์กลางของ Bounding Box (x1, y1, x2, y2)"""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def calculate_distance(p1, p2):
    """คำนวณระยะทางแบบ Euclidean ระหว่างจุด 2 จุด"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
# -----------------------------

@app.route('/detect', methods=['POST']) 
def detect_objects():
    if model is None:
        return jsonify({"error": "Model failed to load."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Please send a file named 'image'."}), 400

    file = request.files['image']
    
    try:
        img_bytes = file.read()
        print(f"\n--- DEBUG START ---")
        print(f"Received file: {file.filename}")
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"DEBUG Error reading image file: {e}")
        return jsonify({"error": "Invalid image format."}), 400

    # 4. รัน YOLO Inference
    # *** ปรับ conf เป็น 0.8 เพื่อกรองผลลัพธ์ที่มีความมั่นใจต่ำกว่า 80% ***
    MIN_CONFIDENCE = 0.80
    results = model(img, verbose=False, conf=MIN_CONFIDENCE, imgsz=640)

    # 5. เตรียมตัวแปรสำหรับเก็บผลลัพธ์แบบแยกประเภท
    bottle_detections = []
    cap_related_detections = []
    
    # 6. กำหนดคลาสที่ต้องการ
    BOTTLE_CLASS = 'plastic-bottle'
    CAP_RELATED_CLASSES = {'cap', 'no cap'} # ใช้ 'no cap' ตามที่แก้ไข
    
    # 7. ค่า Threshold สำหรับการเชื่อมโยง 
    DISTANCE_THRESHOLD = 100 
    
    # 8. แยกประเภทการตรวจจับ
    all_detected_classes = set()
    
    for r in results:
        if r.boxes:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clss):
                
                # *** ตัวกรองความมั่นใจ (Confidence Filter) ***
                if float(conf) < MIN_CONFIDENCE:
                    continue # ข้ามวัตถุที่มีความมั่นใจต่ำกว่า 80%

                label = model.names[int(cls_id)]
                all_detected_classes.add(label) 
                
                detection_data = {
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "label": label,
                    "center": get_center_point(box) 
                }
                
                if label == BOTTLE_CLASS:
                    bottle_detections.append(detection_data)
                elif label in CAP_RELATED_CLASSES:
                    cap_related_detections.append(detection_data)

    # DEBUG: พิมพ์สรุปผลการตรวจจับเบื้องต้น
    print(f"DEBUG All detected classes (Conf >= {MIN_CONFIDENCE*100}%): {all_detected_classes}")
    print(f"DEBUG Using DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
    print(f"DEBUG Total Bottles found: {len(bottle_detections)}")
    print(f"DEBUG Total Caps/No-Caps found: {len(cap_related_detections)}")
    print("---------------------------------------")

    # 9. การเชื่อมโยงและการกรอง (Association and Filtering)
    final_detections = []
    
    for i, bottle in enumerate(bottle_detections):
        is_verified = False
        
        # ตรวจสอบว่ามีฝา/ไม่มีฝาอยู่ใกล้ขวดนี้หรือไม่
        for cap_item in cap_related_detections:
            distance = calculate_distance(bottle['center'], cap_item['center'])
            
            # DEBUG: แสดงระยะห่างระหว่างขวดและวัตถุที่ใช้ยืนยัน
            # print(f"DEBUG Bottle #{i+1} ({bottle['label']}) vs {cap_item['label']}: Distance={distance:.2f}")

            if distance < DISTANCE_THRESHOLD:
                # พบการเชื่อมโยง: ถือว่าขวดนี้เป็น "ขวดที่ถูกต้อง"
                is_verified = True
                print(f"DEBUG --> Verification SUCCESS for Bottle #{i+1} using {cap_item['label']}")
                
                # 1. เพิ่มขวด (plastic-bottle)
                bottle_to_add = {k: v for k, v in bottle.items() if k != 'center'}
                final_detections.append(bottle_to_add)
                
                # 2. เพิ่มฝา/ไม่มีฝา (cap/no-cap)
                cap_to_add = {k: v for k, v in cap_item.items() if k != 'center'}
                final_detections.append(cap_to_add)

                # สำคัญ: ลบ cap/no-cap ออกจากรายการชั่วคราว
                cap_related_detections.remove(cap_item) 
                
                break # เลิกวน cap/no-cap สำหรับขวดนี้ ไปดูขวดต่อไป

    # 10. ส่งคืนผลลัพธ์ที่ผ่านการยืนยัน
    
    # ลบรายการซ้ำ และส่งคืนผลลัพธ์
    unique_detections = {
        (d['label'], tuple(d['box'])): d 
        for d in final_detections
    }.values()
    
    print("---------------------------------------")
    print(f"DEBUG Filtering Result: SUCCESS (Returning {len(unique_detections)} unique verified detections)")
    print(f"--- DEBUG END ---\n")
    
    return jsonify({"detections": list(unique_detections)})
