from maix import app, camera, display, image, key, time
import cv2
import numpy as np
import math
import itertools


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
FIRST_QUESTION_MODE = True
PIECE_BG_BORDER_SAMPLE = 24
PIECE_BG_DIFF_THRESHOLD = 35.0
PIECE_MASK_OPEN_KERNEL = (3, 3)
PIECE_MASK_CLOSE_KERNEL = (3, 3)
# OPEN 会直接削掉凸出的尖角。背景差分已经比较干净，默认关闭，只保留一次小核 CLOSE 补断口。
PIECE_MASK_OPEN_ITERATIONS = 0
PIECE_MASK_CLOSE_ITERATIONS = 1
PIECE_USE_CONVEX_HULL = True
PIECE_APPROX_EPSILON_RATIO = 0.02
PIECE_APPROX_EPSILON_STEP = 0.005
PIECE_APPROX_EPSILON_MAX = 0.06
PIECE_MIN_EDGE_LENGTH_RATIO = 0.055
PIECE_REFINE_CORNERS_BY_LINES = True
PIECE_LINE_FIT_TRIM_RATIO = 0.18
PIECE_MAX_POINTS = 5
PIECE_MAX_COUNT = 4
PIECE_MIN_AREA_RATIO = 0.002
PIECE_MAX_AREA_RATIO = 0.30
PIECE_BORDER_MARGIN = 8
PIECE_FRAME_MASK_MARGIN = 6
PIECE_MIN_BBOX_SIDE = 8
PIECE_MAX_ASPECT_RATIO = 8.0

FIRST_Q_RECT_W_CM = 10.0
FIRST_Q_RECT_H_CM = 6.0
FIRST_Q_DIAG_A = [2.0, 0.0]
FIRST_Q_DIAG_P = [3.6, 1.2]
FIRST_Q_DIAG_Q = [7.6, 4.2]
FIRST_Q_MATCH_SHAPE_WEIGHT = 1.0
FIRST_Q_MATCH_AREA_WEIGHT = 4.0
FIRST_Q_MATCH_POINT_WEIGHT = 0.08

FIRST_Q_TEMPLATES = [
    {"name": "A", "polygon_cm": [[0.0, 0.0], FIRST_Q_DIAG_A, FIRST_Q_DIAG_P, [0.0, 2.0]]},
    {"name": "B", "polygon_cm": [[0.0, 2.0], FIRST_Q_DIAG_P, FIRST_Q_DIAG_Q, [0.0, 3.0]]},
    {"name": "C", "polygon_cm": [[0.0, 3.0], FIRST_Q_DIAG_Q, [10.0, 6.0], [0.0, 6.0]]},
    {"name": "D", "polygon_cm": [FIRST_Q_DIAG_A, [10.0, 0.0], [10.0, 6.0], FIRST_Q_DIAG_Q, FIRST_Q_DIAG_P]},
]

A4_RATIO_MIN = 1.25
A4_RATIO_MAX = 1.65
ANGLE_MIN = 65.0
ANGLE_MAX = 115.0

PRINT_INTERVAL_MS = 500
SHOW_DEBUG_INFO = False
ENABLE_KEY_EXIT = True

# 显示模式：0 原图 + A4 外框，1 透视图 + 最终碎片，2 色差图，3 raw mask，4 clean mask，5 候选轮廓，6 筛选结果 + 参数。
DISPLAY_MODE = 6
DEBUG_SHOW_ORIGINAL_PROCESS = 0

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

def detect_pieces(frame, corners, matrix):
    if not PIECE_DETECTION_ENABLED:
        return [], [], make_empty_piece_debug()

    frame_h, frame_w = frame.shape[:2]
    frame_area = WARP_W * WARP_H
    detect_mask = make_a4_inner_mask(frame.shape[:2], corners)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    bg_color = estimate_background_lab(lab, detect_mask)
    diff = lab.astype(np.float32) - bg_color.reshape(1, 1, 3)
    distance_map = np.sqrt(np.sum(diff * diff, axis=2))
    raw_mask = np.where(distance_map > PIECE_BG_DIFF_THRESHOLD, 255, 0).astype(np.uint8)
    raw_mask = cv2.bitwise_and(raw_mask, detect_mask)

    mask = raw_mask
    if PIECE_MASK_OPEN_ITERATIONS > 0:
        open_kernel = np.ones(PIECE_MASK_OPEN_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=PIECE_MASK_OPEN_ITERATIONS)
    if PIECE_MASK_CLOSE_ITERATIONS > 0:
        close_kernel = np.ones(PIECE_MASK_CLOSE_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=PIECE_MASK_CLOSE_ITERATIONS)
    mask = cv2.bitwise_and(mask, detect_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    piece_contours = []
    accepted_source_contours = []

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

        boundary = cv2.convexHull(contour) if PIECE_USE_CONVEX_HULL else contour
        perimeter = cv2.arcLength(boundary, True)
        if perimeter <= 0:
            continue

        approx = simplify_piece_polygon(boundary, perimeter)
        if approx is None or len(approx) < 3 or len(approx) > PIECE_MAX_POINTS:
            continue

        approx_warp = cv2.perspectiveTransform(approx.astype(np.float32), matrix).astype(np.int32)
        center_warp = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), matrix)[0][0]
        xw, yw, ww, hw = cv2.boundingRect(approx_warp)
        polygon = approx_warp.reshape(-1, 2).astype(int).tolist()
        cx_warp = int(center_warp[0])
        cy_warp = int(center_warp[1])
        template_scores = first_question_template_scores(approx) if FIRST_QUESTION_MODE else []

        piece = {
            "id": len(pieces),
            "area": int(area),
            "center": [cx_warp, cy_warp],
            "bbox": [int(xw), int(yw), int(ww), int(hw)],
            "side": "left" if cx_warp < WARP_W // 2 else "right",
            "points": len(approx_warp),
            "polygon": polygon,
        }
        if FIRST_QUESTION_MODE:
            piece["template_scores"] = template_scores
        pieces.append(piece)
        piece_contours.append(approx_warp)
        accepted_source_contours.append(contour)

    ranked = sorted(zip(pieces, piece_contours, accepted_source_contours), key=lambda item: item[0]["area"], reverse=True)
    ranked = ranked[:PIECE_MAX_COUNT]
    ranked.sort(key=lambda item: (item[0]["center"][1], item[0]["center"][0]))
    pieces = []
    piece_contours = []
    accepted_source_contours = []
    for piece_id, (piece, contour, source_contour) in enumerate(ranked):
        piece = piece.copy()
        piece["id"] = piece_id
        pieces.append(piece)
        piece_contours.append(contour)
        accepted_source_contours.append(source_contour)

    if FIRST_QUESTION_MODE:
        assign_first_question_templates(pieces)

    piece_debug = {
        "distance_map": cv2.warpPerspective(distance_map.astype(np.float32), matrix, (WARP_W, WARP_H)),
        "raw_mask": cv2.warpPerspective(raw_mask, matrix, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST),
        "clean_mask": cv2.warpPerspective(mask, matrix, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST),
        "all_contours": warp_contours(contours, matrix),
        "accepted_contours": piece_contours,
        "accepted_source_contours": warp_contours(accepted_source_contours, matrix),
        "distance_map_original": distance_map,
        "raw_mask_original": raw_mask,
        "clean_mask_original": mask,
        "all_contours_original": contours,
        "accepted_source_contours_original": accepted_source_contours,
        "bg_color": bg_color,
        "raw_contours_count": len(contours),
        "accepted_count": len(pieces),
    }
    return pieces, piece_contours, piece_debug


def make_empty_piece_debug():
    return {
        "distance_map": None,
        "raw_mask": None,
        "clean_mask": None,
        "all_contours": [],
        "accepted_contours": [],
        "accepted_source_contours": [],
        "distance_map_original": None,
        "raw_mask_original": None,
        "clean_mask_original": None,
        "all_contours_original": [],
        "accepted_source_contours_original": [],
        "bg_color": None,
        "raw_contours_count": 0,
        "accepted_count": 0,
    }


def first_question_template_scores(contour):
    if contour is None or len(contour) < 3:
        return []

    source = normalize_contour_for_match(contour)
    scores = []
    for template_index, template in enumerate(FIRST_Q_TEMPLATES):
        template_contour = template_contour_for_match(template["polygon_cm"])
        score = float(cv2.matchShapes(source, template_contour, cv2.CONTOURS_MATCH_I1, 0.0))
        scores.append({
            "index": template_index,
            "name": template["name"],
            "shape_score": score,
            "points": len(template["polygon_cm"]),
        })
    return scores


def assign_first_question_templates(pieces):
    if not pieces:
        return

    template_ratios = first_question_template_area_ratios()
    total_area = max(1.0, float(sum(piece.get("area", 0) for piece in pieces)))
    piece_count = min(len(pieces), len(FIRST_Q_TEMPLATES))
    best_assignment = None
    best_cost = None

    for template_indices in itertools.permutations(range(len(FIRST_Q_TEMPLATES)), piece_count):
        cost = 0.0
        for piece_index, template_index in enumerate(template_indices):
            piece = pieces[piece_index]
            shape_score = get_template_shape_score(piece, template_index)
            area_ratio = float(piece.get("area", 0)) / total_area
            area_score = abs(area_ratio - template_ratios[template_index])
            point_score = abs(int(piece.get("points", 0)) - len(FIRST_Q_TEMPLATES[template_index]["polygon_cm"]))
            cost += (
                FIRST_Q_MATCH_SHAPE_WEIGHT * shape_score
                + FIRST_Q_MATCH_AREA_WEIGHT * area_score
                + FIRST_Q_MATCH_POINT_WEIGHT * point_score
            )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignment = template_indices

    if best_assignment is None:
        return

    for piece_index, template_index in enumerate(best_assignment):
        piece = pieces[piece_index]
        area_ratio = float(piece.get("area", 0)) / total_area
        area_score = abs(area_ratio - template_ratios[template_index])
        shape_score = get_template_shape_score(piece, template_index)
        template = FIRST_Q_TEMPLATES[template_index]
        piece["template"] = template["name"]
        piece["template_score"] = shape_score + FIRST_Q_MATCH_AREA_WEIGHT * area_score
        piece["template_shape_score"] = shape_score
        piece["template_area_ratio"] = area_ratio
        piece["template_target_area_ratio"] = template_ratios[template_index]
        piece["template_points"] = len(template["polygon_cm"])


def get_template_shape_score(piece, template_index):
    for score in piece.get("template_scores", []):
        if score["index"] == template_index:
            return float(score["shape_score"])
    return 999.0


def first_question_template_area_ratios():
    areas = [polygon_area_cm(template["polygon_cm"]) for template in FIRST_Q_TEMPLATES]
    total = max(1e-6, sum(areas))
    return [area / total for area in areas]


def polygon_area_cm(points_cm):
    points = np.asarray(points_cm, dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def normalize_contour_for_match(contour):
    points = contour.reshape(-1, 2).astype(np.float32)
    center = np.mean(points, axis=0)
    points = points - center
    scale = max(1.0, float(np.max(np.linalg.norm(points, axis=1))))
    points = points / scale * 100.0
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def template_contour_for_match(points_cm):
    points = np.asarray(points_cm, dtype=np.float32)
    center = np.mean(points, axis=0)
    points = points - center
    scale = max(1.0, float(np.max(np.linalg.norm(points, axis=1))))
    points = points / scale * 100.0
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def estimate_background_lab(lab, detect_mask):
    ys, xs = np.where(detect_mask > 0)
    if len(xs) == 0:
        return np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)

    x0 = max(0, int(xs.min()) + PIECE_BG_BORDER_SAMPLE)
    x1 = min(lab.shape[1], int(xs.max()) - PIECE_BG_BORDER_SAMPLE)
    y0 = max(0, int(ys.min()) + PIECE_BG_BORDER_SAMPLE)
    y1 = min(lab.shape[0], int(ys.max()) - PIECE_BG_BORDER_SAMPLE)
    if x1 <= x0 or y1 <= y0:
        samples = lab[detect_mask > 0].reshape(-1, 3)
    else:
        center_mask = np.zeros(detect_mask.shape, dtype=np.uint8)
        center_mask[y0:y1, x0:x1] = 255
        sample_mask = cv2.bitwise_and(detect_mask, center_mask)
        samples = lab[sample_mask > 0].reshape(-1, 3)
        if len(samples) == 0:
            samples = lab[detect_mask > 0].reshape(-1, 3)

    return np.median(samples, axis=0).astype(np.float32)


def make_a4_inner_mask(shape, corners):
    mask = np.zeros(shape, dtype=np.uint8)
    corners_array = np.asarray(corners, dtype=np.float32)
    center = np.mean(corners_array, axis=0)
    inner = center + (corners_array - center) * 0.97
    cv2.fillPoly(mask, [inner.astype(np.int32)], 255)
    if PIECE_FRAME_MASK_MARGIN > 0:
        kernel_size = PIECE_FRAME_MASK_MARGIN * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def warp_contours(contours, matrix):
    warped_contours = []
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        warped = cv2.perspectiveTransform(contour.astype(np.float32), matrix).astype(np.int32)
        warped_contours.append(warped)
    return warped_contours


def simplify_piece_polygon(contour, perimeter):
    epsilon_ratio = PIECE_APPROX_EPSILON_RATIO
    best = None
    while epsilon_ratio <= PIECE_APPROX_EPSILON_MAX:
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) >= 3:
            approx = merge_short_polygon_edges(approx, perimeter)
            if FIRST_QUESTION_MODE and PIECE_REFINE_CORNERS_BY_LINES and 3 <= len(approx) <= PIECE_MAX_POINTS:
                refined = refine_polygon_corners_by_lines(contour, approx)
                if refined is not None:
                    approx = refined
            best = approx
            if 3 <= len(approx) <= PIECE_MAX_POINTS:
                return approx
        epsilon_ratio += PIECE_APPROX_EPSILON_STEP
    return best


def refine_polygon_corners_by_lines(contour, approx):
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    approx_points = approx.reshape(-1, 2).astype(np.float32)
    if len(contour_points) < len(approx_points) or len(approx_points) < 3:
        return approx

    vertex_indices = []
    for point in approx_points:
        distances = np.linalg.norm(contour_points - point, axis=1)
        vertex_indices.append(int(np.argmin(distances)))

    edge_lines = []
    count = len(approx_points)
    for index in range(count):
        start_index = vertex_indices[index]
        end_index = vertex_indices[(index + 1) % count]
        edge_points = contour_points_between(contour_points, start_index, end_index)
        edge_points = trim_edge_points(edge_points)
        line = fit_line(edge_points)
        if line is None:
            return approx
        edge_lines.append(line)

    refined_points = []
    for index in range(count):
        prev_line = edge_lines[(index - 1) % count]
        next_line = edge_lines[index]
        intersection = line_model_intersection(prev_line, next_line)
        if intersection is None:
            return approx

        # 只修正圆角造成的小偏差；交点过远通常意味着分段或拟合出错。
        if np.linalg.norm(intersection - approx_points[index]) > 40.0:
            return approx
        refined_points.append(intersection)

    refined = np.rint(refined_points).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(refined):
        return approx
    return refined


def contour_points_between(points, start_index, end_index):
    if start_index <= end_index:
        return points[start_index:end_index + 1]
    return np.vstack((points[start_index:], points[:end_index + 1]))


def trim_edge_points(points):
    if len(points) <= 4:
        return points
    trim = int(len(points) * PIECE_LINE_FIT_TRIM_RATIO)
    max_trim = (len(points) - 2) // 2
    trim = min(trim, max_trim)
    if trim <= 0:
        return points
    return points[trim:-trim]


def fit_line(points):
    if points is None or len(points) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
    direction = np.array([vx, vy], dtype=np.float32)
    origin = np.array([x0, y0], dtype=np.float32)
    if np.linalg.norm(direction) <= 1e-6:
        return None
    return origin, direction


def line_model_intersection(line_a, line_b):
    origin_a, direction_a = line_a
    origin_b, direction_b = line_b
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(float(denominator)) <= 1e-6:
        return None

    delta = origin_b - origin_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return origin_a + t * direction_a


def merge_short_polygon_edges(approx, perimeter):
    points = approx.reshape(-1, 2).astype(np.float32)
    min_edge_length = perimeter * PIECE_MIN_EDGE_LENGTH_RATIO

    while len(points) > 3:
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        edge_index = int(np.argmin(edge_lengths))
        if edge_lengths[edge_index] >= min_edge_length:
            break

        count = len(points)
        prev_point = points[(edge_index - 1) % count]
        edge_start = points[edge_index]
        edge_end = points[(edge_index + 1) % count]
        next_point = points[(edge_index + 2) % count]
        intersection = line_intersection(prev_point, edge_start, edge_end, next_point)
        if intersection is None:
            break

        # 圆角产生的短边，其相邻直边交点应靠近短边；过远说明两边近似平行，不应强行合并。
        midpoint = (edge_start + edge_end) * 0.5
        if np.linalg.norm(intersection - midpoint) > max(12.0, min_edge_length * 3.0):
            break

        merged = []
        for index in range(count):
            if index == edge_index:
                merged.append(intersection)
            elif index == (edge_index + 1) % count:
                continue
            else:
                merged.append(points[index])
        points = np.asarray(merged, dtype=np.float32)

    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def line_intersection(a0, a1, b0, b1):
    direction_a = a1 - a0
    direction_b = b1 - b0
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(float(denominator)) <= 1e-6:
        return None

    delta = b0 - a0
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return a0 + t * direction_a


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
        label = str(piece["id"])
        if FIRST_QUESTION_MODE and piece.get("template"):
            label += ":" + piece["template"]
        cv2.putText(frame, label, (cx + 5, max(12, cy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)


def draw_piece_debug_info(frame, result, fps, piece_debug):
    draw_text_bg(frame, 8, 24, "A4 OK  FPS:%d" % fps, GREEN, 0.7)
    draw_text_bg(frame, 8, 46, "mode:%d thr:%d" % (DISPLAY_MODE, int(PIECE_BG_DIFF_THRESHOLD)), WHITE, 0.5)
    draw_text_bg(
        frame,
        8,
        66,
        "raw:%d pass:%d" % (piece_debug.get("raw_contours_count", 0), piece_debug.get("accepted_count", 0)),
        WHITE,
        0.5,
    )
    draw_text_bg(
        frame,
        8,
        86,
        "open:%dx%d/%d close:%dx%d/%d" % (
            PIECE_MASK_OPEN_KERNEL[0],
            PIECE_MASK_OPEN_KERNEL[1],
            PIECE_MASK_OPEN_ITERATIONS,
            PIECE_MASK_CLOSE_KERNEL[0],
            PIECE_MASK_CLOSE_KERNEL[1],
            PIECE_MASK_CLOSE_ITERATIONS,
        ),
        WHITE,
        0.5,
    )


def gray_to_bgr(gray):
    if gray is None:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def distance_map_to_view(distance_map):
    if distance_map is None:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    view_gray = np.clip(distance_map * (255.0 / max(1.0, PIECE_BG_DIFF_THRESHOLD * 2.0)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(view_gray, cv2.COLOR_GRAY2BGR)


def make_debug_view(frame, debug):
    if DEBUG_VIEW_MODE == 1:
        return cv2.cvtColor(debug.get("canny"), cv2.COLOR_GRAY2BGR)
    if DEBUG_VIEW_MODE == 2:
        return cv2.cvtColor(debug.get("edges"), cv2.COLOR_GRAY2BGR)
    return frame.copy()


def make_display_view(frame, result, fps, a4_warp=None, pieces=None, piece_contours=None, piece_debug=None):
    piece_debug = piece_debug or make_empty_piece_debug()

    if result.get("status") and DISPLAY_MODE == 2:
        key_name = "distance_map_original" if DEBUG_SHOW_ORIGINAL_PROCESS else "distance_map"
        view = distance_map_to_view(piece_debug.get(key_name))
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if result.get("status") and DISPLAY_MODE == 3:
        key_name = "raw_mask_original" if DEBUG_SHOW_ORIGINAL_PROCESS else "raw_mask"
        view = gray_to_bgr(piece_debug.get(key_name))
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if result.get("status") and DISPLAY_MODE == 4:
        key_name = "clean_mask_original" if DEBUG_SHOW_ORIGINAL_PROCESS else "clean_mask"
        view = gray_to_bgr(piece_debug.get(key_name))
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if result.get("status") and DISPLAY_MODE == 5:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            view = frame.copy()
            cv2.drawContours(view, piece_debug.get("all_contours_original", []), -1, RED, 1)
        else:
            view = a4_warp.copy() if a4_warp is not None else warp_a4(frame, result["corners"])[0]
            cv2.drawContours(view, piece_debug.get("all_contours", []), -1, RED, 1)
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if result.get("status") and DISPLAY_MODE == 6:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            view = frame.copy()
            cv2.drawContours(view, piece_debug.get("accepted_source_contours_original", []), -1, RED, 1)
            draw_original_result(view, result, fps)
        else:
            view = a4_warp.copy() if a4_warp is not None else warp_a4(frame, result["corners"])[0]
            cv2.drawContours(view, piece_debug.get("accepted_source_contours", []), -1, RED, 1)
            draw_pieces(view, pieces or [], piece_contours or [])
            draw_warp_result(view, result, fps)
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

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
        piece_debug = make_empty_piece_debug()
        if result.get("status"):
            a4_warp, matrix, _ = warp_a4(frame, result["corners"])
            pieces, piece_contours, piece_debug = detect_pieces(frame, result["corners"], matrix)
            result = result.copy()
            result["pieces_count"] = len(pieces)
            result["pieces"] = pieces

        now = time.ticks_ms()
        frame_count += 1
        if now - last_fps_ms >= 1000:
            fps = frame_count
            frame_count = 0
            last_fps_ms = now

        view = make_display_view(frame, result, fps, a4_warp, pieces, piece_contours, piece_debug)

        if now - last_print_ms >= PRINT_INTERVAL_MS:
            print(result)
            last_print_ms = now

        disp.show(image.cv2image(view, bgr=True, copy=True))

    if key_obj:
        del key_obj


if __name__ == "__main__":
    main()
