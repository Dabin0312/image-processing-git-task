📸 Image Processing Git TaskOpenCV를 활용한 HSV 기반 객체 검출과 HuggingFace 데이터셋을 이용한 이미지 전처리 파이프라인 구현 프로젝트입니다.🛠 RequirementsPython 3.xVirtual Environment (Recommended)Bash# 환경 설정 및 라이브러리 설치
pip install -r requirements.txt
🔴 1. Red Color Detection (HSV)HSV 색 공간을 활용하여 이미지 내의 빨간색 영역을 추출하고 마스킹 처리를 수행합니다.🚀 ExecutionBashpython src/main.py --input sample.jpg --show
📁 Outputsoutput/mask_red.png: 빨간색 영역만 추출된 바이너리 마스크output/result_red.png: 원본 이미지에 마스크를 적용한 결과물🖼 2. Image Preprocessing PipelineHuggingFace (ethz/food101) 데이터셋을 활용하여 딥러닝 모델 학습에 적합한 전처리 파이프라인을 구축했습니다.⚙️ Preprocessing StepsStandardization: $224 \times 224$ 리사이징 및 Grayscale 변환Normalization: $[0, 1]$ 정규화 후 다시 $[0, 255]$ (uint8) 변환Denoising: 가우시안 블러(Gaussian Blur) 적용Filtering (Advanced):평균 밝기 기준 너무 어두운 이미지 제외Otsu Threshold 및 컨투어(Contour) 분석을 통해 객체 크기가 너무 작은 이미지 필터링Augmentation (Optional): Flip, Rotation, Brightness 조정🚀 ExecutionBash# 기본 전처리 이미지 5장 생성
python image_preprocessing.py --num_samples 5

# 데이터 증강(Augmentation) 포함 실행
python image_preprocessing.py --num_samples 5 --save_aug
📁 Outputs결과물은 preprocessed_samples/ 디렉토리에 저장됩니다.sample_n_preprocessed.png: 전처리가 완료된 이미지sample_n_aug_k.png: 데이터 증강이 적용된 이미지💡 수정 포인트 (Tip)이모지 활용: 섹션별로 관련 이모지를 넣어 시각적 구분감을 주었습니다.단계별 리스트: 전처리 과정을 숫자로 나열하여 파이프라인의 흐름을 한눈에 파악하게 했습니다.코드 블럭 최적화: 실행 명령어와 파일 경로를 명확히 구분했습니다.



