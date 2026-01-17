import os
import json
import cv2
from ultralytics import YOLO
from collections import Counter

INPUT_DIR = "week3/inputs"
OUT_STATS = "week3/outputs/stats"
MODEL_NAME = "yolov8n.pt"

def main():
    os.makedirs(OUT_STATS, exist_ok=True)

    model = YOLO(MODEL_NAME)

    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("[ERROR] week3/inputs 폴더에 이미지가 없습니다.")
        return

    class_counter = Counter()
    per_image_summary = []

    for fname in image_files:
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)

        results = model(img)[0]
        names = results.names

        detected_classes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detected_classes.append(names[cls_id])

        class_counter.update(detected_classes)

        per_image_summary.append({
            "file": fname,
            "num_objects": len(detected_classes),
            "classes": detected_classes
        })

    with open(os.path.join(OUT_STATS, "per_image_summary.json"), "w", encoding="utf-8") as f:
        json.dump(per_image_summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_STATS, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(class_counter, f, ensure_ascii=False, indent=2)

    print("[OK] saved stats:", OUT_STATS)
    print("Top classes:", class_counter.most_common(10))

if __name__ == "__main__":
    main()
