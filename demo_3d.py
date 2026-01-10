import cv2
from processing_3d import generate_depth_map

image = cv2.imread("sample.jpg")
if image is None:
    raise FileNotFoundError("sample.jpg를 찾을 수 없습니다.")

depth_map = generate_depth_map(image)
cv2.imwrite("depth_map.png", depth_map)
print("Saved: depth_map.png")
