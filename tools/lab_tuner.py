import argparse
import glob
import os

import cv2
import numpy as np


WINDOW = "LAB tuner"

def nothing(_value):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("前景阈值", WINDOW, 35, 160, nothing)
    cv2.createTrackbar("亮度权重x100", WINDOW, 25, 200, nothing)
    cv2.createTrackbar("绿红权重x100", WINDOW, 100, 200, nothing)
    cv2.createTrackbar("蓝黄权重x100", WINDOW, 100, 200, nothing)
    cv2.createTrackbar("背景采样边宽", WINDOW, 24, 120, nothing)
    cv2.createTrackbar("开运算次数", WINDOW, 0, 5, nothing)
    cv2.createTrackbar("闭运算次数", WINDOW, 0, 5, nothing)


def get_params():
    threshold = cv2.getTrackbarPos("前景阈值", WINDOW)
    l_weight = cv2.getTrackbarPos("亮度权重x100", WINDOW) / 100.0
    a_weight = cv2.getTrackbarPos("绿红权重x100", WINDOW) / 100.0
    b_weight = cv2.getTrackbarPos("蓝黄权重x100", WINDOW) / 100.0
    border_sample = cv2.getTrackbarPos("背景采样边宽", WINDOW)
    open_iter = cv2.getTrackbarPos("开运算次数", WINDOW)
    close_iter = cv2.getTrackbarPos("闭运算次数", WINDOW)
    return threshold, l_weight, a_weight, b_weight, border_sample, open_iter, close_iter


def estimate_background_lab(lab, border_sample):
    h, w = lab.shape[:2]
    border_sample = max(1, min(border_sample, h // 3, w // 3))
    samples = np.vstack(
        [
            lab[:border_sample, :, :].reshape(-1, 3),
            lab[h - border_sample :, :, :].reshape(-1, 3),
            lab[:, :border_sample, :].reshape(-1, 3),
            lab[:, w - border_sample :, :].reshape(-1, 3),
        ]
    )
    return np.median(samples, axis=0).astype(np.float32)


def make_mask(frame, params):
    threshold, l_weight, a_weight, b_weight, border_sample, open_iter, close_iter = params
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    bg_color = estimate_background_lab(lab, border_sample)
    diff = lab.astype(np.float32) - bg_color.reshape(1, 1, 3)
    weights = np.array([l_weight, a_weight, b_weight], dtype=np.float32).reshape(1, 1, 3)
    distance_map = np.sqrt(np.sum(diff * diff * weights, axis=2))
    raw_mask = np.where(distance_map > threshold, 255, 0).astype(np.uint8)

    mask = raw_mask
    if open_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    view_gray = np.clip(distance_map * (255.0 / max(1.0, threshold * 2.0)), 0, 255).astype(np.uint8)
    return bg_color, view_gray, raw_mask, mask


def overlay_mask(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def fit_to_height(image, height):
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def build_view(frame, distance_gray, raw_mask, clean_mask):
    distance_bgr = cv2.cvtColor(distance_gray, cv2.COLOR_GRAY2BGR)
    raw_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    clean_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    overlay = overlay_mask(frame, raw_mask)
    height = 300
    row1 = np.hstack([fit_to_height(frame, height), fit_to_height(distance_bgr, height)])
    row2 = np.hstack([fit_to_height(overlay, height), fit_to_height(raw_bgr, height)])
    row3 = np.hstack([fit_to_height(clean_bgr, height), fit_to_height(frame, height)])
    return np.vstack([row1, row2, row3])


def image_files(path):
    if os.path.isdir(path):
        files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(path, pattern)))
        return sorted(files)
    return [path]


def main():
    parser = argparse.ArgumentParser(description="Tune LAB color-distance thresholds.")
    parser.add_argument("path", nargs="?", default="test_photo", help="Image file or image directory.")
    args = parser.parse_args()

    files = image_files(args.path)
    if not files:
        raise SystemExit("No images found: %s" % args.path)

    create_trackbars()
    index = 0
    last_params = None
    while True:
        frame = cv2.imread(files[index])
        if frame is None:
            raise SystemExit("Failed to read image: %s" % files[index])

        params = get_params()
        bg_color, distance_gray, raw_mask, clean_mask = make_mask(frame, params)
        view = build_view(frame, distance_gray, raw_mask, clean_mask)
        cv2.putText(
            view,
            os.path.basename(files[index]),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, view)

        if params != last_params:
            threshold, l_weight, a_weight, b_weight, border_sample, open_iter, close_iter = params
            print(
                "LAB thr=%d weights=(%.2f,%.2f,%.2f) border=%d open=%d close=%d bg=(%.1f,%.1f,%.1f)"
                % (
                    threshold,
                    l_weight,
                    a_weight,
                    b_weight,
                    border_sample,
                    open_iter,
                    close_iter,
                    bg_color[0],
                    bg_color[1],
                    bg_color[2],
                )
            )
            last_params = params

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), ord("d")):
            index = (index + 1) % len(files)
        if key in (ord("p"), ord("a")):
            index = (index - 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
