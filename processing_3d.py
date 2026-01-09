import cv2
import numpy as np

def generate_depth_map(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map

def depth_to_point_cloud(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)

    points_3d = np.dstack((X.astype(np.float32), Y.astype(np.float32), Z))
    return points_3d
