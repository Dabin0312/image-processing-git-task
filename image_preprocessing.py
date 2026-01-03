import argparse
import os
import random

import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def preprocess_rgb_to_gray_u8(img_rgb: np.ndarray) -> np.ndarray:
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
    out = []
    out.append(cv2.flip(gray_u8, 1))  # 좌우 반전

    h, w = gray_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)  # 회전
    out.append(cv2.warpAffine(gray_u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))

    brighter = np.clip(gray_u8.astype(np.int16) + 25, 0, 255).astype(np.uint8)  # 밝기 변화
    out.append(brighter)
    return out


def is_too_dark(gray_u8: np.ndarray, mean_thresh: float = 40.0) -> bool:
    return float(gray_u8.mean()) < mean_thresh


def is_object_too_small(gray_u8: np.ndarray, min_area_ratio: float = 0.01) -> bool:
    h, w = gray_u8.shape[:2]
    total = h * w
    _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def max_contour_area(bin_img):
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return float(max(cv2.contourArea(c) for c in contours))

    a1 = max_contour_area(bw)
    a2 = max_contour_area(255 - bw)
    max_area = max(a1, a2)

    return (max_area / total) < min_area_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ethz/food101")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--out_dir", default="preprocessed_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_aug", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)

    saved = 0
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    for i in tqdm(idxs):
        item = ds[i]
        pil_img: Image.Image = item["image"]
        img_rgb = np.array(pil_img.convert("RGB"))

        proc = preprocess_rgb_to_gray_u8(img_rgb)

        # advanced filtering
        if is_too_dark(proc):
            continue
        if is_object_too_small(proc):
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


if __name__ == "__main__":
    main()
