import math

import cv2
import numpy as np

import config


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


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


def quad_side_lengths(corners):
    tl, tr, br, bl = corners
    return [
        distance(tl, tr),
        distance(tr, br),
        distance(br, bl),
        distance(bl, tl),
    ]


def quad_aspect_ratio(corners):
    top, right, bottom, left = quad_side_lengths(corners)
    width = (top + bottom) * 0.5
    height = (left + right) * 0.5
    if min(width, height) <= 1e-6:
        return 0.0
    return max(width, height) / min(width, height)


def draw_status(frame, text, ok):
    color = config.COLOR_SUCCESS if ok else config.COLOR_FAIL
    cv2.rectangle(frame, (8, 8), (260, 42), config.COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_debug_info(frame, debug):
    if not config.SHOW_DEBUG_INFO:
        return frame

    h = frame.shape[0]
    lines = [
        f"contours: {debug.get('contours', 0)}",
        f"candidates: {debug.get('candidates', 0)}",
        f"area: {debug.get('best_area_ratio', 0.0):.2f}",
        f"ratio: {debug.get('best_aspect_ratio', 0.0):.2f}",
    ]

    angles = debug.get("best_angles") or []
    if angles:
        lines.append("angles: " + ",".join(f"{angle:.0f}" for angle in angles))

    x = 10
    y0 = max(70, h - 22 * len(lines) - 10)
    cv2.rectangle(frame, (x - 4, y0 - 18), (270, y0 + 22 * len(lines) - 2), config.COLOR_TEXT_BG, -1)
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)
    return frame


def _to_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _make_tile(image, title):
    tile = _to_bgr(image)
    if tile is None:
        tile = np.zeros((config.DEBUG_TILE_HEIGHT, config.DEBUG_TILE_WIDTH, 3), dtype=np.uint8)

    tile = cv2.resize(tile, (config.DEBUG_TILE_WIDTH, config.DEBUG_TILE_HEIGHT))
    cv2.rectangle(tile, (0, 0), (config.DEBUG_TILE_WIDTH, 26), config.COLOR_TEXT_BG, -1)
    cv2.putText(tile, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)
    return tile


def make_process_debug_view(frame, result_view, debug):
    # 2x3 拼图：从左到右能看到图像被一步步处理成最终轮廓的过程。
    tiles = [
        _make_tile(frame, "1 original"),
        _make_tile(debug.get("gray"), "2 gray"),
        _make_tile(debug.get("blur"), "3 gaussian blur"),
        _make_tile(debug.get("canny"), "4 canny"),
        _make_tile(debug.get("edges"), "5 morph close"),
        _make_tile(result_view, "6 result"),
    ]
    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:])
    return np.vstack([top, bottom])


def draw_a4_result(frame, result):
    if not result.get("status"):
        draw_status(frame, "A4 NOT FOUND", False)
        return frame

    corners = np.asarray(result["corners"], dtype=np.int32)
    cv2.polylines(frame, [corners], True, config.COLOR_SUCCESS, 3)

    labels = ["TL", "TR", "BR", "BL"]
    for label, (x, y) in zip(labels, corners):
        cv2.circle(frame, (int(x), int(y)), 6, config.COLOR_CORNER, -1)
        cv2.putText(frame, label, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_CORNER, 2)

    draw_status(frame, "A4 DETECTED", True)
    return frame
