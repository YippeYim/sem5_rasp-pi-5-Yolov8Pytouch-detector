# ... (โค้ดส่วน import และ app setup)

# 7. ค่า Threshold สำหรับการเชื่อมโยง (อาจต้องปรับเพิ่ม)
# หากขวดยังไม่ถูกยืนยัน ลองเพิ่มค่านี้เป็น 120 หรือ 150
DISTANCE_THRESHOLD = 100 
    
# 8. แยกประเภทการตรวจจับ
all_detected_classes = set()
    
# 6. กำหนดคลาสที่ต้องการ (แก้ไขตรงนี้)
BOTTLE_CLASS = 'plastic-bottle'
# *** แก้ไข: ใช้ 'no cap' แทน 'no-cap' เพื่อให้ตรงกับ log ที่พบ ***
CAP_RELATED_CLASSES = {'cap', 'no cap'} 
    
for r in results:
# ... (โค้ดส่วนการวนลูป)

    for box, conf, cls_id in zip(boxes, confs, clss):
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
            # ตอนนี้ 'no cap' ที่มีช่องว่างจะถูกเก็บเข้าในรายการนี้แล้ว
            cap_related_detections.append(detection_data)

# ... (โค้ดส่วน Debug และการเชื่อมโยง)
