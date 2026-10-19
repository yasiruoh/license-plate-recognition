REPO = 'Koushim/yolov8-license-plate-detection'
FILENAME = 'best.pt'
ALLOWED = []

from functools import lru_cache
from collections import Counter
from src.validation import threshold_value

@lru_cache(maxsize=1)
def load_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    return YOLO(hf_hub_download(REPO, FILENAME))

def predict(image, threshold=0.35):
    import numpy as np
    import cv2
    model = load_model()
    result = model.predict(image, conf=threshold_value(threshold), verbose=False)[0]
    rows = []
    for box in result.boxes:
        label = str(result.names[int(box.cls.item())])
        if ALLOWED and label not in ALLOWED:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        row = {"label": label, "confidence": round(float(box.conf.item()), 6),
               "box": [x1, y1, x2, y2]}
        rows.append(row)
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    for row in rows:
        x1,y1,x2,y2 = row["box"]
        h,w = canvas.shape[:2]
        crop = canvas[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        row["text"] = ""
        row["ocr_confidence"] = None
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            tokens = load_ocr().readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", detail=1)
            row["text"] = " ".join(t[1] for t in tokens)
            if tokens:
                row["ocr_confidence"] = round(sum(float(t[2]) for t in tokens)/len(tokens),6)
    for row in rows:
        x1, y1, x2, y2 = row["box"]
        text = row.get("text", "") or row["label"]
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (30,180,120), 2)
        cv2.putText(canvas, f"{text} {row['confidence']:.2f}", (x1,max(18,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,180,120), 2)
    return {"detections": rows, "counts": dict(Counter(r["label"] for r in rows)),
            "total": len(rows)}, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

@lru_cache(maxsize=1)
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)
