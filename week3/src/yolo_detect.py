import os
import cv2
from ultralytics import YOLO

INPUT_DIR = "week3/inputs"
OUTPUT_DIR = "week3/outputs/vis"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")  # 가볍고 빠른 모델

    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("[ERROR] week3/inputs 폴더에 이미지가 없습니다.")
        return

    for fname in image_files:
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)

        results = model(img)[0]     # 탐지 결과
        plotted = results.plot()    # 박스 포함 이미지

        save_path = os.path.join(OUTPUT_DIR, f"det_{fname}")
        cv2.imwrite(save_path, plotted)

        print(f"[OK] saved: {save_path}")

if __name__ == "__main__":
    main()
