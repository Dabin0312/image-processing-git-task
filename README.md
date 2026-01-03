## Image Processing Git Task

OpenCV HSV 기반 **빨간색 검출 및 필터링**과 HuggingFace 데이터셋 기반 **이미지 전처리 파이프라인**을 구현했습니다.

---

## Requirements

- Python 3.x  
- (권장) 가상환경 사용

설치:
bash
pip install -r requirements.txt



\## 1) Red Color Detection (HSV)

Run
python src/main.py --input sample.jpg --show

Output
output/mask_red.png
output/result_red.png


\## Preprocessing (HuggingFace)
Dataset
ethz/food101

Steps
\- Resize to 224×224
\- Convert to grayscale
\- Normalize (0~1) then save as uint8 (0~255)
\- Gaussian blur for noise reduction
\- (Advanced) Filter out too-dark images by mean brightness
\- (Advanced) Filter out too-small objects using Otsu threshold + largest contour area ratio
\- (Optional) Data augmentation: flip / rotate / brightness change

Run (save 5 preprocessed images)
python image_preprocessing.py --num_samples 5

Run (save augmented images too)
python image_preprocessing.py --num_samples 5 --save_aug


Outputs
preprocessed_samples/sample_1_preprocessed.png ~ sample_5_preprocessed.png
preprocessed_samples/sample_1_aug1.png ... *_aug*.png



