# 📸 Image Processing Git Task

OpenCV를 활용한 **HSV 기반 객체 검출**과 HuggingFace 데이터셋을 이용한 **이미지 전처리 파이프라인** 구현 프로젝트입니다.

---

## 🛠 Requirements

* **Python 3.x**
* **Virtual Environment (Recommended)**

```bash
# 환경 설정 및 라이브러리 설치
pip install -r requirements.txt


## 🔴 1. Red Color Detection (HSV)
HSV 색 공간을 활용하여 이미지 내의 빨간색 영역을 추출하고 마스킹 처리를 수행합니다.
python src/main.py --input sample.jpg --show

## 🖼 2. Image Preprocessing Pipeline
HuggingFace (ethz/food101) 데이터셋을 활용하여 딥러닝 모델 학습에 적합한 전처리 파이프라인을 구축했습니다.

⚙️ Preprocessing Steps
Standardization: 224 × 224 리사이징 및 Grayscale 변환

Normalization: [0, 1] 정규화 후 다시 [0, 255] (uint8) 변환

Denoising: 가우시안 블러(Gaussian Blur) 적용

Filtering (Advanced):

평균 밝기 기준 너무 어두운 이미지 제외

Otsu Threshold 및 컨투어(Contour) 분석을 통해 객체 크기가 너무 작은 이미지 필터링

Augmentation (Optional): Flip, Rotation, Brightness 조정

##🚀 Execution
# 기본 전처리 이미지 5장 생성
python image_preprocessing.py --num_samples 5

# 데이터 증강(Augmentation) 포함 실행
python image_preprocessing.py --num_samples 5 --save_aug


## 📁 Outputs
결과물은 preprocessed_samples/ 디렉토리에 저장됩니다.
sample_n_preprocessed.png: 전처리가 완료된 이미지
sample_n_aug_k.png: 데이터 증강이 적용된 이미지













