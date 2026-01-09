import numpy as np
import pytest

from processing_3d import generate_depth_map, depth_to_point_cloud

def test_generate_depth_map_valid_input():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)

    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."

def test_generate_depth_map_none_input():
    with pytest.raises(ValueError, match="입력된 이미지가 없습니다."):
        generate_depth_map(None)

def test_depth_to_point_cloud_shape():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    points_3d = depth_to_point_cloud(image)

    assert points_3d.shape == (50, 60, 3)
    assert points_3d.dtype == np.float32
