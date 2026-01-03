import argparse
import os
import random

import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def preprocess_rgb_to_gray_u8(img_rgb: np.ndarray) -> np.ndarray:
    """Resize(224x224) -> Grayscale -> Normalize(0~1) -> Blur -> uint8(0~255)"""
    # 1) resize (224x224)
    img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # 2) grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3) normalize (0~1)
    norm = gray.astype(np.float32) / 255.0

    # 4) blur (noise reduction)
    norm_blur = cv2.GaussianBlur(norm, (5, 5), 0)

    # save-friendly uint8
    out = np.clip(norm_blur * 255.0, 0, 255).astype(np.uint8)
    return out


def augment(gray_u8: np.ndarray) -> list[np.ndarray]:
    """Augmentation: flip / rotate / brightness change"""
    out = []

    # 좌우 반전
    out.append(cv2.flip(gray_u8, 1))

    # 회전(10도)
    h, w = gray_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    out.append(
        cv2.warpAffine(
            gray_u8,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    )

    # 밝기 변화
    brighter = np.clip(gray_u8.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    out.append(brighter)

    return out


def is_too_dark(gray_u8: np.ndarray, mean_thresh: float) -> bool:
    """Too dark outlier detection by mean brightness"""
    return float(gray_u8.mean()) < mean_thresh


def is_object_too_small(gray_u8: np.ndarray, min_area_ratio: float) -> bool:
    """
    Too small object outlier detection:
    - Otsu threshold -> find contours -> largest contour area / total area
    - If ratio is too small, treat as outlier
    """
    h, w = gray_u8.shape[:2]
    total = h * w

    _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def max_contour_area(bin_img: np.ndarray) -> float:
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return float(max(cv2.contourArea(c) for c in contours))

    # 전경이 흰색/검정색으로 나뉘는 케이스 모두 대비
    a1 = max_contour_area(bw)
    a2 = max_contour_area(255 - bw)
    max_area = max(a1, a2)

    return (max_area / total) < min_area_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ethz/food101", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="train/validation/test")
    parser.add_argument("--num_samples", type=int, default=5, help="number of images to save")
    parser.add_argument("--out_dir", default="preprocessed_samples", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_aug", action="store_true", help="also save augmented images")

    # Advanced filtering params
    parser.add_argument(
        "--mean_thresh",
        type=float,
        default=40.0,
        help="remove images whose mean brightness is below this threshold (0~255)",
    )
    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=0.01,
        help="remove images whose largest object area ratio is below this (0~1)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)

    saved = 0
    dark_cnt = 0
    small_cnt = 0

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    for i in tqdm(idxs):
        item = ds[i]
        pil_img: Image.Image = item["image"]
        img_rgb = np.array(pil_img.convert("RGB"))

        proc = preprocess_rgb_to_gray_u8(img_rgb)

        # advanced filtering
        if is_too_dark(proc, mean_thresh=args.mean_thresh):
            dark_cnt += 1
            continue
        if is_object_too_small(proc, min_area_ratio=args.min_area_ratio):
            small_cnt += 1
            continue

        saved += 1
        cv2.imwrite(os.path.join(args.out_dir, f"sample_{saved}_preprocessed.png"), proc)

        if args.save_aug:
            for j, a in enumerate(augment(proc), start=1):
                cv2.imwrite(os.path.join(args.out_dir, f"sample_{saved}_aug{j}.png"), a)

        if saved >= args.num_samples:
            break

    if saved < args.num_samples:
        print(f"[WARN] only saved {saved} images. Try lowering thresholds or changing split.")
    else:
        print(f"[OK] saved {saved} images to {args.out_dir}/")

    print(f"filtered: too_dark={dark_cnt}, too_small={small_cnt}")


if __name__ == "__main__":
    main()

