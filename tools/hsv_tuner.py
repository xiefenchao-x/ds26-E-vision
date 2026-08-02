import argparse
import glob
import os

import cv2
import numpy as np


WINDOW = "HSV tuner"

def nothing(_value):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("绿背景H下限", WINDOW, 35, 179, nothing)
    cv2.createTrackbar("绿背景H上限", WINDOW, 95, 179, nothing)
    cv2.createTrackbar("绿背景S下限", WINDOW, 40, 255, nothing)
    cv2.createTrackbar("绿背景S上限", WINDOW, 255, 255, nothing)
    cv2.createTrackbar("绿背景V下限", WINDOW, 40, 255, nothing)
    cv2.createTrackbar("绿背景V上限", WINDOW, 255, 255, nothing)
    cv2.createTrackbar("开运算次数", WINDOW, 0, 5, nothing)
    cv2.createTrackbar("闭运算次数", WINDOW, 0, 5, nothing)
    cv2.createTrackbar("反选成碎片", WINDOW, 1, 1, nothing)


def get_params():
    h_low = cv2.getTrackbarPos("绿背景H下限", WINDOW)
    h_high = cv2.getTrackbarPos("绿背景H上限", WINDOW)
    s_low = cv2.getTrackbarPos("绿背景S下限", WINDOW)
    s_high = cv2.getTrackbarPos("绿背景S上限", WINDOW)
    v_low = cv2.getTrackbarPos("绿背景V下限", WINDOW)
    v_high = cv2.getTrackbarPos("绿背景V上限", WINDOW)
    open_iter = cv2.getTrackbarPos("开运算次数", WINDOW)
    close_iter = cv2.getTrackbarPos("闭运算次数", WINDOW)
    invert = cv2.getTrackbarPos("反选成碎片", WINDOW)
    return h_low, h_high, s_low, s_high, v_low, v_high, open_iter, close_iter, invert


def make_mask(frame, params):
    h_low, h_high, s_low, s_high, v_low, v_high, open_iter, close_iter, invert = params
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([h_low, s_low, v_low], dtype=np.uint8)
    high = np.array([h_high, s_high, v_high], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, low, high)

    if open_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    piece_mask = cv2.bitwise_not(green_mask) if invert else green_mask
    return green_mask, piece_mask


def overlay_mask(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def fit_to_height(image, height):
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def build_view(frame, green_mask, piece_mask):
    green_bgr = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    piece_bgr = cv2.cvtColor(piece_mask, cv2.COLOR_GRAY2BGR)
    overlay = overlay_mask(frame, piece_mask)
    height = 320
    row1 = np.hstack([fit_to_height(frame, height), fit_to_height(overlay, height)])
    row2 = np.hstack([fit_to_height(green_bgr, height), fit_to_height(piece_bgr, height)])
    return np.vstack([row1, row2])


def image_files(path):
    if os.path.isdir(path):
        files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(path, pattern)))
        return sorted(files)
    return [path]


def main():
    parser = argparse.ArgumentParser(description="Tune HSV thresholds for green A4 background.")
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
        green_mask, piece_mask = make_mask(frame, params)
        view = build_view(frame, green_mask, piece_mask)
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
            h_low, h_high, s_low, s_high, v_low, v_high, open_iter, close_iter, invert = params
            print(
                "HSV low=(%d,%d,%d) high=(%d,%d,%d) open=%d close=%d invert=%d"
                % (h_low, s_low, v_low, h_high, s_high, v_high, open_iter, close_iter, invert)
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
