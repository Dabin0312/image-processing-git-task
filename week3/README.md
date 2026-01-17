
# 📌 Week 3 - Object Detection & Pattern Analysis (YOLOv8)

## 🎯 1. Goal
This project performs **object detection** using an AI model (**YOLOv8**) and analyzes detected object patterns.

🔍 What we do:
- Detect objects in images using **YOLOv8**
- Visualize bounding boxes with **OpenCV**
- Analyze class frequency and number of objects per image

---

## 🛠️ 2. Environment
Requirements:
- Python 3.x  
- ultralytics (YOLOv8)  
- opencv-python  
- numpy  
- matplotlib  

📌 Install libraries:
pip install ultralytics opencv-python matplotlib numpy

---

## 📂 3. Folder Structure
```text
week3/
├─ src/
│  ├─ yolo_detect.py
│  └─ analyze_results.py
├─ inputs/
│  ├─ img01.jpg
│  ├─ img02.jpg
│  ├─ img03.jpg
│  ├─ img04.jpg
│  └─ img05.jpg
├─ outputs/
│  ├─ vis/
│  │  ├─ det_img01.jpg
│  │  ├─ det_img02.jpg
│  │  ├─ det_img03.jpg
│  │  ├─ det_img04.jpg
│  │  └─ det_img05.jpg
│  └─ stats/
│     ├─ class_counts.json
│     └─ per_image_summary.json
└─ README.md
```

---

## ▶️ 4. How to Run

🖼️ 4-1) Object Detection (Visualization)
python week3/src/yolo_detect.py

Output:
week3/outputs/vis/det_*.jpg


📊 4-2) Pattern Analysis (Statistics)

Output:
week3/outputs/stats/class_counts.json
week3/outputs/stats/per_image_summary.json

---

## 📌 5. Result Summary

5-1) Total Detected Class Counts (class_counts.json)
- 👤 person: **4**
- 🪑 chair: **3**
- 🍽️ dining table: **3**
- 🔪 knife: **2**
- 🐶 dog: **1**
- 🥣 bowl: **1**
- 📖 book: **1**
- 🎂 cake: **1**

5-2) Pattern Interpretation (Insights)
- 👤 **person** was detected the most → indoor/lifestyle scenes were common.
- 🪑 **chair** and 🍽️ **dining table** appeared frequently → furniture-based indoor environment.
- 🍽️ Food-related objects (**knife, bowl, cake**) appeared → meal-related scenes were included.





---

## 🏁 6. Conclusion
YOLOv8 successfully detected objects in **5 images**, and results were visualized using OpenCV.  
Class frequency analysis helped identify object patterns across the dataset.


## 🚀 7. Improvement Ideas
- 📌 Use more images for better pattern reliability
- 🎯 Apply fine-tuning to improve detection accuracy




