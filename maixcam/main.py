from maix import app, camera, display, image, key, time
import cv2
import numpy as np
import math


# =========================
# Config: MaixCAM2 OpenCV A4
# =========================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WARP_W = 594
WARP_H = 420

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

MAX_LOST_FRAMES = 5

PIECE_DETECTION_ENABLED = True
PIECE_CANNY_LOW = 40
PIECE_CANNY_HIGH = 120
PIECE_EDGE_CLOSE_KERNEL = (5, 5)
PIECE_EDGE_CLOSE_ITERATIONS = 2
PIECE_APPROX_EPSILON_RATIO = 0.015
PIECE_MIN_AREA_RATIO = 0.002
PIECE_MAX_AREA_RATIO = 0.30
PIECE_BORDER_MARGIN = 8
PIECE_MIN_BBOX_SIDE = 8
PIECE_MAX_ASPECT_RATIO = 8.0

A4_RATIO_MIN = 1.25
A4_RATIO_MAX = 1.65
ANGLE_MIN = 65.0
ANGLE_MAX = 115.0

PRINT_INTERVAL_MS = 500
SHOW_DEBUG_INFO = False
ENABLE_KEY_EXIT = True

# 显示模式：0 原图 + A4 外框，1 透视 A4 + 标准中线。
DISPLAY_MODE = 1

# 调试视图保留给现场排查，常规阶段 2 显示由 DISPLAY_MODE 控制。
DEBUG_VIEW_MODE = 0

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)


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


def standard_midline():
    x = WARP_W // 2
    return [[x, 0], [x, WARP_H - 1]]


def warp_a4(frame, corners):
    src = np.float32(corners)
    dst = np.float32(
        [
            [0, 0],
            [WARP_W - 1, 0],
            [WARP_W - 1, WARP_H - 1],
            [0, WARP_H - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))
    return warped, matrix, inverse_matrix


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
        corners = best.astype(int).tolist()
        return {
            "status": True,
            "corners": corners,
            "warp_size": [WARP_W, WARP_H],
            "midline": standard_midline(),
        }

    @staticmethod
    def preprocess(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
        canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        kernel = np.ones(MORPH_CLOSE_KERNEL, np.uint8)
        edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)
        return gray, blur, canny, edges


class A4ResultStabilizer:
    def __init__(self, max_lost_frames):
        self.max_lost_frames = max_lost_frames
        self.last_result = {"status": False}
        self.lost_frames = 0

    def update(self, result):
        if result.get("status"):
            self.last_result = result.copy()
            self.last_result["stable"] = True
            self.last_result["lost_frames"] = 0
            self.lost_frames = 0
            return self.last_result

        if self.last_result.get("status") and self.lost_frames < self.max_lost_frames:
            self.lost_frames += 1
            stable_result = self.last_result.copy()
            stable_result["stable"] = True
            stable_result["held"] = True
            stable_result["lost_frames"] = self.lost_frames
            return stable_result

        self.last_result = {"status": False}
        self.lost_frames = 0
        return {"status": False}


# =========================
# Puzzle piece detector
# =========================

def detect_pieces(a4_warp):
    if not PIECE_DETECTION_ENABLED:
        return [], []

    frame_h, frame_w = a4_warp.shape[:2]
    frame_area = frame_h * frame_w
    gray = cv2.cvtColor(a4_warp, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, PIECE_CANNY_LOW, PIECE_CANNY_HIGH)
    kernel = np.ones(PIECE_EDGE_CLOSE_KERNEL, np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=PIECE_EDGE_CLOSE_ITERATIONS)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    piece_contours = []

    min_area = frame_area * PIECE_MIN_AREA_RATIO
    max_area = frame_area * PIECE_MAX_AREA_RATIO
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-6:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        if w < PIECE_MIN_BBOX_SIDE or h < PIECE_MIN_BBOX_SIDE:
            continue
        if (
            x <= PIECE_BORDER_MARGIN
            or y <= PIECE_BORDER_MARGIN
            or x + w >= frame_w - PIECE_BORDER_MARGIN
            or y + h >= frame_h - PIECE_BORDER_MARGIN
        ):
            continue

        bbox_ratio = max(w, h) / max(1.0, float(min(w, h)))
        if bbox_ratio > PIECE_MAX_ASPECT_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, PIECE_APPROX_EPSILON_RATIO * perimeter, True)
        if len(approx) < 3:
            continue

        polygon = approx.reshape(-1, 2).astype(int).tolist()

        piece = {
            "id": len(pieces),
            "area": int(area),
            "center": [cx, cy],
            "bbox": [int(x), int(y), int(w), int(h)],
            "side": "left" if cx < WARP_W // 2 else "right",
            "points": len(approx),
            "polygon": polygon,
        }
        pieces.append(piece)
        piece_contours.append(approx)

    return pieces, piece_contours


# =========================
# Draw / app
# =========================

def draw_text_bg(frame, x, y, text, color, scale=0.55):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x - 3, y - h - 4), (x + w + 3, y + 4), BLACK, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_original_result(frame, result, fps):
    if result.get("status"):
        corners = np.asarray(result["corners"], dtype=np.int32)
        cv2.polylines(frame, [corners], True, GREEN, 2)
        for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
            cv2.circle(frame, (int(x), int(y)), 4, YELLOW, -1)
            cv2.putText(frame, label, (int(x) + 4, max(12, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
        draw_text_bg(frame, 8, 24, "A4 OK  FPS:%d" % fps, GREEN, 0.7)
    else:
        draw_text_bg(frame, 8, 24, "A4 NO  FPS:%d" % fps, RED, 0.7)


def draw_warp_result(frame, result, fps):
    if result.get("status"):
        top_mid, bottom_mid = result["midline"]
        cv2.line(frame, tuple(top_mid), tuple(bottom_mid), YELLOW, 2)
        draw_text_bg(frame, 8, 24, "A4 OK  FPS:%d" % fps, GREEN, 0.7)
    else:
        draw_text_bg(frame, 8, 24, "A4 NO  FPS:%d" % fps, RED, 0.7)


def draw_pieces(frame, pieces, piece_contours):
    for piece, contour in zip(pieces, piece_contours):
        cv2.drawContours(frame, [contour], -1, BLUE, 2)
        cx, cy = piece["center"]
        cv2.circle(frame, (cx, cy), 4, YELLOW, -1)
        cv2.putText(frame, str(piece["id"]), (cx + 5, max(12, cy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)


def make_debug_view(frame, debug):
    if DEBUG_VIEW_MODE == 1:
        return cv2.cvtColor(debug.get("canny"), cv2.COLOR_GRAY2BGR)
    if DEBUG_VIEW_MODE == 2:
        return cv2.cvtColor(debug.get("edges"), cv2.COLOR_GRAY2BGR)
    return frame.copy()


def make_display_view(frame, result, fps, a4_warp=None, pieces=None, piece_contours=None):
    if DISPLAY_MODE == 1 and result.get("status"):
        view = a4_warp.copy() if a4_warp is not None else warp_a4(frame, result["corners"])[0]
        draw_pieces(view, pieces or [], piece_contours or [])
        draw_warp_result(view, result, fps)
        return view

    view = frame.copy()
    if DISPLAY_MODE == 1:
        draw_warp_result(view, result, fps)
    else:
        draw_original_result(view, result, fps)
    return view


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
    stabilizer = A4ResultStabilizer(MAX_LOST_FRAMES)
    key_obj = key.Key(on_key) if ENABLE_KEY_EXIT else None
    last_print_ms = time.ticks_ms()
    last_fps_ms = last_print_ms
    frame_count = 0
    fps = 0

    while not app.need_exit():
        maix_img = cam.read()
        frame = image.image2cv(maix_img, ensure_bgr=True, copy=True)

        raw_result = detector.detect(frame)
        result = stabilizer.update(raw_result)
        a4_warp = None
        pieces = []
        piece_contours = []
        if result.get("status"):
            a4_warp, _, _ = warp_a4(frame, result["corners"])
            pieces, piece_contours = detect_pieces(a4_warp)
            result = result.copy()
            result["pieces_count"] = len(pieces)
            result["pieces"] = pieces

        now = time.ticks_ms()
        frame_count += 1
        if now - last_fps_ms >= 1000:
            fps = frame_count
            frame_count = 0
            last_fps_ms = now

        view = make_display_view(frame, result, fps, a4_warp, pieces, piece_contours)

        if now - last_print_ms >= PRINT_INTERVAL_MS:
            print(result)
            last_print_ms = now

        disp.show(image.cv2image(view, bgr=True, copy=True))

    if key_obj:
        del key_obj


if __name__ == "__main__":
    main()
