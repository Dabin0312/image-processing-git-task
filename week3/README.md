\# Week 3 - Object Detection \& Pattern Analysis (YOLOv8)



\## 1. Goal

This project performs object detection using an AI model (YOLOv8) and analyzes detected object patterns.

\- Detect objects in images using YOLOv8

\- Visualize bounding boxes with OpenCV

\- Analyze class frequency and objects per image



---



\## 2. Environment

\- Python 3.x

\- ultralytics (YOLOv8)

\- opencv-python

\- numpy



Install libraries:

```bash

pip install ultralytics opencv-python matplotlib numpy





\## 3. Folder Structure

week3/

&nbsp;├─ src/

&nbsp;│   ├─ yolo\_detect.py

&nbsp;│   └─ analyze\_results.py

&nbsp;├─ inputs/

&nbsp;│   ├─ img01.jpg

&nbsp;│   ├─ img02.jpg

&nbsp;│   ├─ img03.jpg

&nbsp;│   ├─ img04.jpg

&nbsp;│   └─ img05.jpg

&nbsp;├─ outputs/

&nbsp;│   ├─ vis/

&nbsp;│   │   ├─ det\_img01.jpg

&nbsp;│   │   ├─ det\_img02.jpg

&nbsp;│   │   ├─ det\_img03.jpg

&nbsp;│   │   ├─ det\_img04.jpg

&nbsp;│   │   └─ det\_img05.jpg

&nbsp;│   └─ stats/

&nbsp;│       ├─ class\_counts.json

&nbsp;│       └─ per\_image\_summary.json

&nbsp;└─ README.md







