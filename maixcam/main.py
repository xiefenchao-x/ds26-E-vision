from maix import app, camera, display, image, key, time
import cv2
import numpy as np
import math


# =========================
# Config: MaixCAM2 OpenCV A4
# =========================

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_CLOSE_KERNEL = (5, 5)
MORPH_CLOSE_ITERATIONS = 1

APPROX_EPSILON_RATIO = 0.025
MIN_AREA_RATIO = 0.07
MAX_AREA_RATIO = 0.65

# 防止把中间黑线分出的半张纸当成 A4。
# 如果同一帧里存在更大的 A4 外轮廓，即使它没有拟合成四边形，小候选也会被过滤。
CANDIDATE_MIN_RELATIVE_AREA = 0.65

A4_RATIO_MIN = 1.25
A4_RATIO_MAX = 1.65
ANGLE_MIN = 65.0
ANGLE_MAX = 115.0

PRINT_INTERVAL_MS = 500
SHOW_DEBUG_INFO = True
ENABLE_KEY_EXIT = True

# 显示模式：0 原图结果，1 Canny，2 闭运算边缘。现场调试时可手动切换。
DEBUG_VIEW_MODE = 0

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# =========================
# Geometry
# =========================

def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def distance(point_a, point_b):
    return float(np.linalg.norm(np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)))


def angle_at(prev_point, center_point, next_point):
    vec_a = np.asarray(prev_point, dtype=np.float32) - np.asarray(center_point, dtype=np.float32)
    vec_b = np.asarray(next_point, dtype=np.float32) - np.asarray(center_point, dtype=np.float32)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm <= 1e-6:
        return 0.0
    cosine = float(np.dot(vec_a, vec_b) / norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def quad_angles(corners):
    tl, tr, br, bl = corners
    return [
        angle_at(bl, tl, tr),
        angle_at(tl, tr, br),
        angle_at(tr, br, bl),
        angle_at(br, bl, tl),
    ]


def quad_aspect_ratio(corners):
    tl, tr, br, bl = corners
    top = distance(tl, tr)
    right = distance(tr, br)
    bottom = distance(bl, br)
    left = distance(tl, bl)
    width = (top + bottom) * 0.5
    height = (left + right) * 0.5
    if min(width, height) <= 1e-6:
        return 0.0
    return max(width, height) / min(width, height)


# =========================
# OpenCV A4 detector
# =========================

class A4Detector:
    def __init__(self):
        self.debug = {}

    def detect(self, frame):
        frame_area = frame.shape[0] * frame.shape[1]
        gray, blur, canny, edges = self.preprocess(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        best_ratio = 0.0
        best_angles = []
        candidate_count = 0
        area_pass = 0
        quad_pass = 0
        ratio_pass = 0
        angle_pass = 0
        max_area_ratio = 0.0
        max_area_pass_ratio = 0.0
        max_approx_points = 0
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / frame_area
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio
            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                continue
            area_pass += 1
            if area_ratio > max_area_pass_ratio:
                max_area_pass_ratio = area_ratio

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, True)
            if len(approx) > max_approx_points:
                max_approx_points = len(approx)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad_pass += 1

            corners = order_points(approx.reshape(4, 2))
            ratio = quad_aspect_ratio(corners)
            if not (A4_RATIO_MIN <= ratio <= A4_RATIO_MAX):
                continue
            ratio_pass += 1

            angles = quad_angles(corners)
            if not all(ANGLE_MIN <= angle <= ANGLE_MAX for angle in angles):
                continue
            angle_pass += 1

            candidates.append((area, area_ratio, corners, ratio, angles))

        min_relative_area = max_area_pass_ratio * CANDIDATE_MIN_RELATIVE_AREA
        for area, area_ratio, corners, ratio, angles in candidates:
            if area_ratio < min_relative_area:
                continue

            candidate_count += 1
            if area > best_area:
                best = corners
                best_area = area
                best_ratio = ratio
                best_angles = angles

        self.debug = {
            "contours": len(contours),
            "candidates": candidate_count,
            "area_pass": area_pass,
            "quad_pass": quad_pass,
            "ratio_pass": ratio_pass,
            "angle_pass": angle_pass,
            "max_area_ratio": max_area_ratio,
            "max_area_pass_ratio": max_area_pass_ratio,
            "min_relative_area": min_relative_area,
            "max_approx_points": max_approx_points,
            "best_area_ratio": best_area / frame_area if frame_area > 0 else 0.0,
            "best_aspect_ratio": best_ratio,
            "best_angles": best_angles,
            "gray": gray,
            "canny": canny,
            "edges": edges,
        }

        if best is None:
            return {"status": False}
        return {"status": True, "corners": best.astype(int).tolist()}

    @staticmethod
    def preprocess(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
        canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        kernel = np.ones(MORPH_CLOSE_KERNEL, np.uint8)
        edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)
        return gray, blur, canny, edges


# =========================
# Draw / app
# =========================

def draw_text_bg(frame, x, y, text, color, scale=0.55):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x - 3, y - h - 4), (x + w + 3, y + 4), BLACK, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_result(frame, result, debug):
    if result.get("status"):
        corners = np.asarray(result["corners"], dtype=np.int32)
        cv2.polylines(frame, [corners], True, GREEN, 2)
        for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
            cv2.circle(frame, (int(x), int(y)), 4, YELLOW, -1)
            cv2.putText(frame, label, (int(x) + 4, max(12, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
        draw_text_bg(frame, 8, 24, "A4 DETECTED", GREEN, 0.7)
    else:
        draw_text_bg(frame, 8, 24, "A4 NOT FOUND", RED, 0.7)

    if SHOW_DEBUG_INFO:
        draw_text_bg(frame, 8, 48, "cont:%d cand:%d" % (debug.get("contours", 0), debug.get("candidates", 0)), WHITE)
        draw_text_bg(frame, 8, 68, "pass a/q/r/g:%d/%d/%d/%d" % (
            debug.get("area_pass", 0),
            debug.get("quad_pass", 0),
            debug.get("ratio_pass", 0),
            debug.get("angle_pass", 0),
        ), WHITE)
        draw_text_bg(frame, 8, 88, "max_area:%.2f pts:%d" % (debug.get("max_area_ratio", 0), debug.get("max_approx_points", 0)), WHITE)
        draw_text_bg(frame, 8, 108, "max_pass:%.2f rel_min:%.2f" % (debug.get("max_area_pass_ratio", 0), debug.get("min_relative_area", 0)), WHITE)
        draw_text_bg(frame, 8, 128, "area:%.2f ratio:%.2f" % (debug.get("best_area_ratio", 0), debug.get("best_aspect_ratio", 0)), WHITE)
        angles = debug.get("best_angles") or []
        if angles:
            draw_text_bg(frame, 8, 148, "ang:%d,%d,%d,%d" % tuple([int(a) for a in angles]), WHITE)


def make_debug_view(frame, debug):
    if DEBUG_VIEW_MODE == 1:
        return cv2.cvtColor(debug.get("canny"), cv2.COLOR_GRAY2BGR)
    if DEBUG_VIEW_MODE == 2:
        return cv2.cvtColor(debug.get("edges"), cv2.COLOR_GRAY2BGR)
    return frame.copy()


def on_key(key_id, state):
    if ENABLE_KEY_EXIT and state == key.State.KEY_LONG_PRESSED:
        app.set_exit_flag(True)


def create_camera():
    try:
        return camera.Camera(FRAME_WIDTH, FRAME_HEIGHT, image.Format.FMT_BGR888)
    except Exception:
        return camera.Camera(FRAME_WIDTH, FRAME_HEIGHT)


def main():
    cam = create_camera()
    try:
        cam.skip_frames(30)
    except Exception:
        pass

    disp = display.Display()
    detector = A4Detector()
    key_obj = key.Key(on_key) if ENABLE_KEY_EXIT else None
    last_print_ms = time.ticks_ms()

    while not app.need_exit():
        maix_img = cam.read()
        frame = image.image2cv(maix_img, ensure_bgr=True, copy=True)

        result = detector.detect(frame)
        view = make_debug_view(frame, detector.debug)
        draw_result(view, result, detector.debug)

        now = time.ticks_ms()
        if now - last_print_ms >= PRINT_INTERVAL_MS:
            print(result)
            last_print_ms = now

        disp.show(image.cv2image(view, bgr=True, copy=True))

    if key_obj:
        del key_obj


if __name__ == "__main__":
    main()
