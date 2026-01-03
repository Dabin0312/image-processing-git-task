# Image Processing Git Task

OpenCV HSV 기반 빨간색 검출 및 필터링

## Run

python src/main.py --input sample.jpg --show

## Output

output/mask\_red.png, output/result\_red.png



\## Preprocessing (HuggingFace)



Dataset: ethz/food101



Steps

\- Resize to 224x224

\- Convert to grayscale

\- Normalize (0~1) then save as uint8 (0~255)

\- Gaussian blur for noise reduction

\- (Advanced) Filter out too-dark images by mean brightness

\- (Advanced) Filter out too-small objects using Otsu threshold + largest contour area ratio



Run

python image\_preprocessing.py --num\_samples 5



Outputs

\- preprocessed\_samples/sample\_1\_preprocessed.png ... sample\_5\_preprocessed.png



