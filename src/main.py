import argparse
import os
import cv2
import numpy as np


def detect_red_hsv(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return mask, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input image path")
    parser.add_argument("--out_dir", default="output", help="output directory")
    parser.add_argument("--show", action="store_true", help="show windows")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    mask, result = detect_red_hsv(image)

    os.makedirs(args.out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.out_dir, "mask_red.png"), mask)
    cv2.imwrite(os.path.join(args.out_dir, "result_red.png"), result)

    print("[OK] Saved output/mask_red.png, output/result_red.png")

    if args.show:
        cv2.imshow("Original", image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Red Filtered", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
