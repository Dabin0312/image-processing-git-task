# 📸 Image Processing Git Task

OpenCV를 활용한 **HSV 기반 객체 검출(1차)** 과 HuggingFace 데이터셋을 이용한 **이미지 전처리 파이프라인(1차)**,  
그리고 pytest 기반 **Unit Test + 2D → 3D(Depth Map) 변환(2차)** 를 포함한 실습 프로젝트입니다.

---

## 🛠 Requirements

- Python 3.x
- (권장) Virtual Environment

### 설치
pip install -r requirements.txt


## 🔴 1. Red Color Detection (HSV)
HSV 색 공간을 활용하여 이미지 내의 빨간색 영역을 추출하고 마스킹 처리를 수행합니다.
python src/main.py --input sample.jpg --show

출력
output/mask_red.png
output/result_red.png


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

Optional filtering params (추가 알고리즘):
- --mean_thresh (default 40.0), --min_area_ratio (default 0.01)


## 🚀 Execution
기본 전처리 이미지 5장 생성
python image_preprocessing.py --num_samples 5

데이터 증강(Augmentation) 포함 실행
python image_preprocessing.py --num_samples 5 --save_aug


## 📁 Outputs
결과물은 preprocessed_samples/ 디렉토리에 저장됩니다.
sample_n_preprocessed.png: 전처리가 완료된 이미지
sample_n_aug_k.png: 데이터 증강이 적용된 이미지



## 🧪 Unit Test 작성 및 코드 검증 (pytest)
pytest를 활용해 2D→3D 변환 코드의 정상 동작/예외 처리/출력 shape를 검증합니다.

테스트파일; test_3d_processing.py

정상 입력 시 Depth Map 생성 결과의 type/shape 확인
입력이 None일 때 ValueError 예외 처리 확인
3D 변환 결과(point cloud)의 shape (H, W, 3) 및 dtype 확인

실행
pytest test_3d_processing.py


## 🧊 2D → 3D 변환 (Depth Map / Point Cloud 개념)
OpenCV + NumPy로 2D 이미지를 기반으로 가상 Depth Map을 만들고, 이를 바탕으로 3D 좌표(X,Y,Z) 형태의 포인트를 생성합니다.

구현파일
processing_3d.py
generate_depth_map(image)
depth_to_point_cloud(image)

실
python demo_3d.py


## 🖼 2D → 3D 변환 결과 (Depth Map)
입력 이미지(sample.jpg)를 grayscale로 변환한 뒤 JET colormap을 적용해 가상 Depth Map을 생성했습니다.

![Depth Map](depth_map.png)














