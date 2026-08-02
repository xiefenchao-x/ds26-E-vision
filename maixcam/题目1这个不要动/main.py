from maix import app, camera, display, image, key, time
try:
    from maix import uart
except Exception:
    uart = None
import cv2
import numpy as np
import math
import itertools
import struct


# =========================
# Config: MaixCAM2 OpenCV A4
# =========================
# 第一题主配置区：完成摄像头采集、A4 纸透视矫正、碎片识别、模板匹配和坐标输出。

# 摄像头输入分辨率，单位像素；分辨率越高识别更细，但处理速度更慢。
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# A4 纸透视矫正后的标准画布大小，单位像素。
# 程序检测到 A4 四角后，会把纸面拉正到这个尺寸，后续所有碎片坐标都基于它。
WARP_W = 594
WARP_H = 420

# A4 纸真实尺寸，cm 用于题目尺寸描述，mm 用于机械坐标/串口输出。
A4_W_CM = 29.7
A4_H_CM = 21.0
A4_W_MM = A4_W_CM * 10.0
A4_H_MM = A4_H_CM * 10.0

# A4 外框边缘检测预处理参数。
GAUSSIAN_KERNEL = (5, 5)       # 高斯模糊核，先降噪，避免 Canny 检出太多杂边。
CANNY_LOW = 50                 # Canny 低阈值，影响弱边缘保留程度。
CANNY_HIGH = 150               # Canny 高阈值，影响强边缘判定。
MORPH_CLOSE_KERNEL = (5, 5)    # 闭运算核，用来连接断开的 A4 纸边缘。
MORPH_CLOSE_ITERATIONS = 1     # 闭运算次数，过大会让不该连接的轮廓粘在一起。

# A4 候选轮廓筛选参数。
APPROX_EPSILON_RATIO = 0.025   # 四边形拟合精度比例，越大轮廓越容易被简化。
MIN_AREA_RATIO = 0.07          # A4 候选最小面积占整帧比例，过滤小噪声。
MAX_AREA_RATIO = 0.65          # A4 候选最大面积占整帧比例，过滤异常大区域。

# 防止把中间黑线分出的半张纸当成 A4。
# 如果同一帧里存在更大的 A4 外轮廓，即使它没有拟合成四边形，小候选也会被过滤。
CANDIDATE_MIN_RELATIVE_AREA = 0.65

# A4 外框允许连续丢失的帧数；短暂遮挡/抖动时不立即清空检测状态。
MAX_LOST_FRAMES = 5

# 碎片识别和第一题模式开关。
PIECE_DETECTION_ENABLED = True     # 是否启用碎片检测；False 时只显示/检测 A4 外框。
FIRST_QUESTION_MODE = True         # 是否使用第一题固定 10 cm x 6 cm 模板匹配流程。
PIECE_MASK_METHOD = "black"        # 碎片分割方式；black 表示基于黑色背景和碎片亮度/颜色差分。
PIECE_PROCESS_A4_ROI = True        # 是否只处理透视后的 A4 区域，减少画面外干扰。

# 背景采样和 Lab 色差参数。
PIECE_BG_BORDER_SAMPLE = 24        # 从 A4 边缘向内采样背景的宽度，单位像素。
PIECE_BG_DIFF_THRESHOLD = 35.0     # 与背景色差超过该阈值才认为可能是碎片。
PIECE_L_DIFF_WEIGHT = 0.25         # Lab 亮度 L 通道权重，较低可减小阴影/光照影响。
PIECE_A_DIFF_WEIGHT = 1.0          # Lab a 通道权重，主要描述红绿方向差异。
PIECE_B_DIFF_WEIGHT = 1.0          # Lab b 通道权重，主要描述黄蓝方向差异。

# HSV 绿色范围；用于识别绿色碎片或绿色纸面特征。
PIECE_GREEN_H_LOW = 51             # H 色相下限。
PIECE_GREEN_H_HIGH = 91            # H 色相上限。
PIECE_GREEN_S_LOW = 52             # S 饱和度下限，太低说明颜色偏灰。
PIECE_GREEN_S_HIGH = 255           # S 饱和度上限。
PIECE_GREEN_V_LOW = 75             # V 亮度下限。
PIECE_GREEN_V_HIGH = 255           # V 亮度上限。
PIECE_HSV_DIFF_THRESHOLD = 35.0    # HSV 差分阈值，用于辅助判断目标和背景差异。
PIECE_HSV_DIFF_BLUR_KERNEL = (5, 5)# HSV 差分图模糊核，减少小噪点。

# 黑色背景分割参数；现场是黑底时，用 V 通道亮度把碎片从背景中提出来。
PIECE_BLACK_V_MIN_THRESHOLD = 120  # V 通道绝对亮度阈值，亮于它才可能是碎片。
PIECE_BLACK_V_OFFSET = 85          # 相对背景亮度偏移，适应不同灯光环境。
PIECE_BLACK_BLUR_KERNEL = (3, 3)   # 黑底分割前的模糊核。

# 碎片 mask 清理参数。
PIECE_MASK_MEDIAN_KERNEL = 3       # 中值滤波核，去椒盐噪声。
PIECE_MASK_OPEN_KERNEL = (3, 3)    # 开运算核，用于去小白点。
PIECE_MASK_CLOSE_KERNEL = (3, 3)   # 闭运算核，用于补小断口。
# OPEN 会直接削掉凸出的尖角。背景差分已经比较干净，默认关闭，只保留一次小核 CLOSE 补断口。
PIECE_MASK_OPEN_ITERATIONS = 0
PIECE_MASK_CLOSE_ITERATIONS = 0
PIECE_USE_CLEAN_MASK_FOR_CONTOURS = False # 是否用清理后的 mask 找轮廓；False 可保留更多原始边缘细节。

# 粘连碎片拆分参数；碎片贴在一起时可开启，但可能误拆单块。
PIECE_SPLIT_TOUCHING_ENABLED = False
PIECE_SPLIT_ERODE_KERNEL = (3, 3)      # 腐蚀核大小，用于把粘连区域分开。
PIECE_SPLIT_ERODE_ITERATIONS = 3       # 腐蚀次数，越大拆分越强，也越容易破坏形状。

# 碎片轮廓拟合与过滤参数。
PIECE_USE_CONVEX_HULL = True           # 是否先取凸包，减少轮廓凹坑/毛刺影响。
PIECE_APPROX_EPSILON_RATIO = 0.008     # 多边形拟合初始精度比例，越小保留顶点越多。
PIECE_APPROX_EPSILON_STEP = 0.003      # 拟合失败时逐步放宽的步长。
PIECE_APPROX_EPSILON_MAX = 0.035       # 最大拟合精度比例，防止过度简化。
PIECE_MIN_EDGE_LENGTH_RATIO = 0.025    # 最短边相对 A4 尺寸的比例，过滤毛刺短边。
PIECE_REFINE_CORNERS_BY_LINES = True   # 是否用直线拟合交点细化角点位置。
PIECE_LINE_FIT_TRIM_RATIO = 0.18       # 拟合边线时裁掉两端比例，减少角点附近噪声干扰。
PIECE_MAX_POINTS = 5                   # 单个碎片最多角点数；第一题模板一般是 4 或 5 点。
PIECE_MAX_COUNT = 4                    # 第一题最多识别 4 块碎片。
PIECE_MIN_AREA_RATIO = 0.002           # 碎片最小面积占 A4 区域比例，过滤小噪声。
PIECE_MAX_AREA_RATIO = 0.30            # 碎片最大面积占 A4 区域比例，过滤误检大块。
PIECE_BORDER_MARGIN = 8                # 过滤贴近 A4 边框的干扰轮廓，单位像素。
PIECE_FRAME_MASK_MARGIN = 6            # 生成整帧 mask 时保留的边缘安全距离。
PIECE_MIN_BBOX_SIDE = 8                # 外接框最小边长，过滤极小轮廓。
PIECE_MAX_ASPECT_RATIO = 8.0           # 外接框最大长宽比，过滤细长线状噪声。
PIECE_TARGET_SHAPE_EXPAND_SCALE = 1.3 # 拼好后的目标显示/匹配放大倍数。

# 第一题目标矩形和摆放参数，单位厘米。
FIRST_Q_RECT_W_CM = 10.0               # 目标矩形宽度。
FIRST_Q_RECT_H_CM = 6.0                # 目标矩形高度。
FIRST_Q_TARGET_SIDE = "auto"           # 目标区域放置侧；auto 表示程序自动选择。
FIRST_Q_TARGET_ORIENTATION = "portrait"# 目标布局方向；portrait 表示竖向显示/输出。
FIRST_Q_PLACE_MARGIN_CM = 0.0          # 目标放置外边距，单位厘米。

# 第一题官方切割模板关键点，坐标原点是 10 cm x 6 cm 矩形左上角。
FIRST_Q_DIAG_A = [2.0, 0.0]            # 模板中的 A 点，位于上边。
FIRST_Q_DIAG_P = [3.6, 1.2]            # 模板中的内部连接点 P。
FIRST_Q_DIAG_Q = [7.6, 4.2]            # 模板中的内部连接点 Q。
FIRST_Q_MATCH_SHAPE_WEIGHT = 1.0       # 模板匹配中形状相似度权重。
FIRST_Q_MATCH_AREA_WEIGHT = 4.0        # 模板匹配中面积比例权重。
FIRST_Q_MATCH_POINT_WEIGHT = 0.08      # 模板匹配中角点距离权重。
ROTATION_MATCH_SAMPLE_COUNT = 32       # 旋转匹配时采样多少个角度。
ROTATION_MATCH_MAX_CANDIDATES = 12     # 每块碎片最多保留多少个旋转候选。

# 第一题四块标准模板，单位厘米；A/B/C/D 是目标矩形被切开后的四块标准形状。
FIRST_Q_TEMPLATES = [
    {"name": "A", "polygon_cm": [[0.0, 0.0], FIRST_Q_DIAG_A, FIRST_Q_DIAG_P, [0.0, 2.0]]},
    {"name": "B", "polygon_cm": [[0.0, 2.0], FIRST_Q_DIAG_P, FIRST_Q_DIAG_Q, [0.0, 3.0]]},
    {"name": "C", "polygon_cm": [[0.0, 3.0], FIRST_Q_DIAG_Q, [10.0, 6.0], [0.0, 6.0]]},
    {"name": "D", "polygon_cm": [FIRST_Q_DIAG_A, [10.0, 0.0], [10.0, 6.0], FIRST_Q_DIAG_Q, FIRST_Q_DIAG_P]},
]

# A4 外框形状合理性限制。
A4_RATIO_MIN = 1.25                    # A4 长宽比下限，防止非 A4 四边形误检。
A4_RATIO_MAX = 1.65                    # A4 长宽比上限。
ANGLE_MIN = 65.0                       # A4 四个角允许的最小角度，单位度。
ANGLE_MAX = 115.0                      # A4 四个角允许的最大角度，单位度。

# 打印、按键和性能调试参数。
PRINT_INTERVAL_MS = 500                # 终端/串口打印最小间隔，单位毫秒。
PRINT_MOVE_ONLY = True                 # 只在结果变化时打印，避免刷屏。
SHOW_DEBUG_INFO = False                # 是否在画面叠加详细调试信息。
ENABLE_KEY_EXIT = True                 # 是否允许按键退出程序。
SEND_SERIAL_ON_KEY_PRESS = True        # 是否按键确认后才发送串口数据。
PERF_PROFILE = True                    # 是否统计每帧处理耗时。

# 稳定捕获参数；按键后连续采集多帧，剔除离群值后输出稳定结果。
CAPTURE_FRAME_COUNT = 60               # 一次捕获最多处理帧数。
CAPTURE_HOLD_MS = 65000                # 捕获最长等待时间，单位毫秒。
CAPTURE_MIN_VALID_FRAMES = 2           # 至少需要多少帧有效识别结果。
CAPTURE_OUTLIER_A4_CORNER_PX = 25.0    # A4 角点离群阈值，单位像素。
CAPTURE_OUTLIER_CENTER_PX = 35.0       # 碎片中心离群阈值，单位像素。
CAPTURE_OUTLIER_ROTATE_DEG = 25.0      # 旋转角离群阈值，单位度。

# 串口和 G-code 输出参数。
SERIAL_OUTPUT_ENABLED = True           # 是否启用串口输出。
SERIAL_PORT = "/dev/ttyS4"             # MaixCAM 上连接下位机的串口设备。
SERIAL_BAUDRATE = 115200               # 串口波特率。
SERIAL_OUTPUT_FORMAT = "binary"        # 输出格式：gcode、binary、text。
GCODE_FEEDRATE = 3000                  # G-code 加工/下笔进给速度。
GCODE_TRAVEL_FEEDRATE = 5000           # G-code 空走移动速度。
GCODE_PEN_UP_Z = 5.0                   # G-code 抬笔高度。
GCODE_PEN_DOWN_Z = 0.0                 # G-code 下笔高度。
GCODE_ROTATE_AXIS = "A"                # G-code 中使用的旋转轴名称。

# 机械坐标标定点，顺序是标准 A4 透视图里的 TL/TR/BR/BL。
# 默认值表示以 A4 左上角为机械原点，单位 mm；实车标定时改成机械端实测坐标。
MECH_COORD_OUTPUT_ENABLED = True       # 是否输出机械坐标。
MECH_SWAP_XY_FOR_STM32 = True          # 是否交换 X/Y，适配 STM32 端坐标定义。
MECH_COORD_DECIMALS = 0                # 机械坐标输出保留小数位数。
MECH_ROTATE_SCALE = 10.0               # 旋转角缩放倍数，适配下位机协议。
MECH_CALIBRATION_POINTS = [
    [0.0, 0.0],
    [A4_W_MM, 0.0],
    [A4_W_MM, A4_H_MM],
    [0.0, A4_H_MM],
]

# 显示模式：0 原图 + A4 外框，1 透视图 + 最终碎片，2 色差图，3 raw mask，4 clean mask，5 候选轮廓，6 筛选结果 + 参数。
DISPLAY_MODE = 1
# 1 表示调试图直接显示原始摄像头坐标，避免透视插值把绿色窄缝显示成灰边。
DEBUG_SHOW_ORIGINAL_PROCESS = 1

# 调试视图保留给现场排查，常规阶段 2 显示由 DISPLAY_MODE 控制。
DEBUG_VIEW_MODE = 0

# OpenCV 绘图颜色，顺序是 BGR，不是 RGB。
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

# 运行时状态和缓存，避免每帧重复计算固定模板。
capture_requested = False                  # 是否已经请求一次稳定捕获。
first_question_area_ratios_cache = None     # 第一题模板面积比例缓存。
first_question_template_contour_cache = {}  # 第一题模板轮廓缓存。
first_question_target_layout_cache = {}     # 第一题目标布局缓存。
rotation_target_sample_cache = {}           # 旋转匹配采样点缓存。


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


def a4_perspective_matrices(corners):
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
    return matrix, inverse_matrix


def warp_a4(frame, corners):
    matrix, inverse_matrix = a4_perspective_matrices(corners)
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
    process_frame, process_mask, roi = crop_to_mask_roi(frame, detect_mask)
    distance_map, raw_mask, bg_color = make_piece_mask(process_frame, process_mask)

    mask = raw_mask
    if PIECE_MASK_OPEN_ITERATIONS > 0:
        open_kernel = np.ones(PIECE_MASK_OPEN_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=PIECE_MASK_OPEN_ITERATIONS)
    if PIECE_MASK_CLOSE_ITERATIONS > 0:
        close_kernel = np.ones(PIECE_MASK_CLOSE_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=PIECE_MASK_CLOSE_ITERATIONS)
    if PIECE_MASK_OPEN_ITERATIONS > 0 or PIECE_MASK_CLOSE_ITERATIONS > 0:
        mask = cv2.bitwise_and(mask, detect_mask)

    if PIECE_SPLIT_TOUCHING_ENABLED:
        split_mask = split_touching_piece_mask(raw_mask, process_mask)
        contour_mask = split_mask
        contour_mask_name = "split"
    else:
        split_mask = raw_mask
        contour_mask = mask if PIECE_USE_CLEAN_MASK_FOR_CONTOURS else raw_mask
        contour_mask_name = "clean" if PIECE_USE_CLEAN_MASK_FOR_CONTOURS else "raw"
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = offset_contours_from_roi(contours, roi)
    distance_map, raw_mask, mask, split_mask = expand_debug_images_from_roi(
        distance_map,
        raw_mask,
        mask,
        split_mask,
        roi,
        frame.shape[:2],
    )
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
        xw, yw, ww, hw = cv2.boundingRect(approx_warp)
        area_warp = cv2.contourArea(approx_warp)
        polygon = approx_warp.reshape(-1, 2).astype(int).tolist()
        center_warp = contour_centroid(approx_warp)
        if center_warp is None:
            center_warp = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), matrix)[0][0]
            cx_warp = int(center_warp[0])
            cy_warp = int(center_warp[1])
        else:
            cx_warp, cy_warp = center_warp
        expanded_polygon = expand_polygon_about_center(polygon, [cx_warp, cy_warp], PIECE_TARGET_SHAPE_EXPAND_SCALE)
        template_scores = first_question_template_scores(approx_warp) if FIRST_QUESTION_MODE else []

        piece = {
            "id": len(pieces),
            "area": int(area_warp),
            "source_area": int(area),
            "center": [cx_warp, cy_warp],
            "bbox": [int(xw), int(yw), int(ww), int(hw)],
            "side": "left" if cx_warp < WARP_W // 2 else "right",
            "points": len(approx_warp),
            "polygon": polygon,
            "expanded_polygon": expanded_polygon,
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
        attach_first_question_targets(pieces)

    piece_debug = make_piece_debug(
        distance_map,
        raw_mask,
        mask,
        split_mask,
        contours,
        accepted_source_contours,
        piece_contours,
        matrix,
        bg_color,
        contour_mask_name,
        len(pieces),
    )
    return pieces, piece_contours, piece_debug


def make_empty_piece_debug():
    return {
        "distance_map": None,
        "raw_mask": None,
        "clean_mask": None,
        "split_mask": None,
        "all_contours": [],
        "accepted_contours": [],
        "accepted_source_contours": [],
        "distance_map_original": None,
        "raw_mask_original": None,
        "clean_mask_original": None,
        "split_mask_original": None,
        "all_contours_original": [],
        "accepted_source_contours_original": [],
        "bg_color": None,
        "raw_contours_count": 0,
        "accepted_count": 0,
        "contour_mask": "?",
    }


def make_piece_debug(
    distance_map,
    raw_mask,
    clean_mask,
    split_mask,
    all_contours,
    accepted_source_contours,
    accepted_contours,
    matrix,
    bg_color,
    contour_mask_name,
    accepted_count,
):
    debug = make_empty_piece_debug()
    debug["bg_color"] = bg_color
    debug["raw_contours_count"] = len(all_contours)
    debug["accepted_count"] = accepted_count
    debug["contour_mask"] = contour_mask_name

    if DISPLAY_MODE == 2:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            debug["distance_map_original"] = distance_map
        else:
            debug["distance_map"] = cv2.warpPerspective(distance_map.astype(np.float32), matrix, (WARP_W, WARP_H))

    if DISPLAY_MODE == 3:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            debug["raw_mask_original"] = raw_mask
        else:
            debug["raw_mask"] = cv2.warpPerspective(raw_mask, matrix, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST)

    if DISPLAY_MODE == 4:
        if PIECE_SPLIT_TOUCHING_ENABLED:
            if DEBUG_SHOW_ORIGINAL_PROCESS:
                debug["split_mask_original"] = split_mask
            else:
                debug["split_mask"] = cv2.warpPerspective(split_mask, matrix, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST)
        else:
            if DEBUG_SHOW_ORIGINAL_PROCESS:
                debug["clean_mask_original"] = clean_mask
            else:
                debug["clean_mask"] = cv2.warpPerspective(clean_mask, matrix, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST)

    if DISPLAY_MODE == 5:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            debug["all_contours_original"] = all_contours
        else:
            debug["all_contours"] = warp_contours(all_contours, matrix)

    if DISPLAY_MODE == 6:
        if DEBUG_SHOW_ORIGINAL_PROCESS:
            debug["accepted_source_contours_original"] = accepted_source_contours
        else:
            debug["accepted_source_contours"] = warp_contours(accepted_source_contours, matrix)

    debug["accepted_contours"] = accepted_contours
    return debug


def crop_to_mask_roi(frame, detect_mask):
    if not PIECE_PROCESS_A4_ROI:
        return frame, detect_mask, None

    nonzero = cv2.findNonZero(detect_mask)
    if nonzero is None:
        return frame, detect_mask, None

    x, y, w, h = cv2.boundingRect(nonzero)
    return frame[y:y + h, x:x + w], detect_mask[y:y + h, x:x + w], (x, y, w, h)


def offset_contours_from_roi(contours, roi):
    if roi is None:
        return contours

    x, y, _, _ = roi
    offset = np.array([[[x, y]]], dtype=np.int32)
    return [contour + offset for contour in contours]


def expand_debug_images_from_roi(distance_map, raw_mask, clean_mask, split_mask, roi, shape):
    if roi is None or not DEBUG_SHOW_ORIGINAL_PROCESS or DISPLAY_MODE not in (2, 3, 4):
        return distance_map, raw_mask, clean_mask, split_mask

    return (
        expand_roi_image(distance_map, roi, shape, np.float32),
        expand_roi_image(raw_mask, roi, shape, np.uint8),
        expand_roi_image(clean_mask, roi, shape, np.uint8),
        expand_roi_image(split_mask, roi, shape, np.uint8),
    )


def expand_roi_image(image_roi, roi, shape, dtype):
    x, y, w, h = roi
    image = np.zeros(shape, dtype=dtype)
    image[y:y + h, x:x + w] = image_roi
    return image


def make_piece_mask(frame, detect_mask):
    if PIECE_MASK_METHOD == "hsv":
        return make_piece_mask_hsv(frame, detect_mask)
    if PIECE_MASK_METHOD == "black":
        return make_piece_mask_black(frame, detect_mask)
    return make_piece_mask_lab(frame, detect_mask)


def make_piece_mask_hsv(frame, detect_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    distance_map = hsv_outside_green_distance(hsv)
    distance_map = smooth_distance_map(distance_map, detect_mask)
    piece_mask = np.where(distance_map > PIECE_HSV_DIFF_THRESHOLD, 255, 0).astype(np.uint8)
    piece_mask = cv2.bitwise_and(piece_mask, detect_mask)
    piece_mask = smooth_piece_mask(piece_mask, detect_mask)

    if DISPLAY_MODE != 2:
        distance_map = np.zeros(piece_mask.shape, dtype=np.float32)
    bg_color = np.array([PIECE_GREEN_H_LOW, PIECE_GREEN_S_LOW, PIECE_GREEN_V_LOW], dtype=np.float32)
    return distance_map, piece_mask, bg_color


def hsv_outside_green_distance(hsv):
    hsv_float = hsv.astype(np.float32)
    h = hsv_float[:, :, 0]
    s = hsv_float[:, :, 1]
    v = hsv_float[:, :, 2]

    h_dist = np.maximum(np.maximum(PIECE_GREEN_H_LOW - h, h - PIECE_GREEN_H_HIGH), 0.0)
    s_dist = np.maximum(np.maximum(PIECE_GREEN_S_LOW - s, s - PIECE_GREEN_S_HIGH), 0.0)
    v_dist = np.maximum(np.maximum(PIECE_GREEN_V_LOW - v, v - PIECE_GREEN_V_HIGH), 0.0)
    return np.sqrt(h_dist * h_dist + s_dist * s_dist + v_dist * v_dist)


def smooth_distance_map(distance_map, detect_mask):
    distance_map = distance_map * (detect_mask.astype(np.float32) / 255.0)
    if PIECE_HSV_DIFF_BLUR_KERNEL[0] <= 1 or PIECE_HSV_DIFF_BLUR_KERNEL[1] <= 1:
        return distance_map

    blurred = cv2.GaussianBlur(distance_map, PIECE_HSV_DIFF_BLUR_KERNEL, 0)
    return blurred * (detect_mask.astype(np.float32) / 255.0)


def make_piece_mask_black(frame, detect_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    distance_map = hsv[:, :, 2].astype(np.float32)
    distance_map = smooth_black_brightness(distance_map, detect_mask)
    bg_v = estimate_black_background_v(distance_map, detect_mask)
    threshold = max(float(PIECE_BLACK_V_MIN_THRESHOLD), bg_v + float(PIECE_BLACK_V_OFFSET))
    piece_mask = np.where(distance_map > threshold, 255, 0).astype(np.uint8)
    piece_mask = cv2.bitwise_and(piece_mask, detect_mask)
    piece_mask = smooth_piece_mask(piece_mask, detect_mask)

    if DISPLAY_MODE != 2:
        distance_map = np.zeros(piece_mask.shape, dtype=np.float32)
    bg_color = np.array([threshold, bg_v, 0.0], dtype=np.float32)
    return distance_map, piece_mask, bg_color


def estimate_black_background_v(distance_map, detect_mask):
    samples = distance_map[detect_mask > 0]
    if len(samples) == 0:
        return 0.0
    return float(np.percentile(samples, 20))


def smooth_black_brightness(distance_map, detect_mask):
    distance_map = distance_map * (detect_mask.astype(np.float32) / 255.0)
    if PIECE_BLACK_BLUR_KERNEL[0] <= 1 or PIECE_BLACK_BLUR_KERNEL[1] <= 1:
        return distance_map

    blurred = cv2.GaussianBlur(distance_map, PIECE_BLACK_BLUR_KERNEL, 0)
    return blurred * (detect_mask.astype(np.float32) / 255.0)


def make_piece_mask_lab(frame, detect_mask):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    bg_color = estimate_background_lab(lab, detect_mask)
    diff = lab.astype(np.float32) - bg_color.reshape(1, 1, 3)
    distance_weights = np.array(
        [PIECE_L_DIFF_WEIGHT, PIECE_A_DIFF_WEIGHT, PIECE_B_DIFF_WEIGHT],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    distance_map = np.sqrt(np.sum(diff * diff * distance_weights, axis=2))
    raw_mask = np.where(distance_map > PIECE_BG_DIFF_THRESHOLD, 255, 0).astype(np.uint8)
    raw_mask = cv2.bitwise_and(raw_mask, detect_mask)
    raw_mask = smooth_piece_mask(raw_mask, detect_mask)
    return distance_map, raw_mask, bg_color


def smooth_piece_mask(mask, detect_mask):
    if PIECE_MASK_MEDIAN_KERNEL <= 1:
        return mask

    kernel = PIECE_MASK_MEDIAN_KERNEL
    if kernel % 2 == 0:
        kernel += 1
    smoothed = cv2.medianBlur(mask, kernel)
    return cv2.bitwise_and(smoothed, detect_mask)


def split_touching_piece_mask(raw_mask, detect_mask):
    if not PIECE_SPLIT_TOUCHING_ENABLED:
        return raw_mask

    kernel = np.ones(PIECE_SPLIT_ERODE_KERNEL, np.uint8)
    split_mask = cv2.erode(raw_mask, kernel, iterations=PIECE_SPLIT_ERODE_ITERATIONS)
    split_mask = cv2.bitwise_and(split_mask, detect_mask)
    return split_mask


def first_question_template_scores(contour):
    if contour is None or len(contour) < 3:
        return []

    source = normalize_contour_for_match(contour)
    scores = []
    for template_index, template in enumerate(FIRST_Q_TEMPLATES):
        template_contour = cached_template_contour_for_match(template_index)
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

    template_ratios = cached_first_question_template_area_ratios()
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
            area_score = template_area_match_score(area_ratio, template_ratios[template_index])
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
        area_score = template_area_match_score(area_ratio, template_ratios[template_index])
        shape_score = get_template_shape_score(piece, template_index)
        template = FIRST_Q_TEMPLATES[template_index]
        piece["template"] = template["name"]
        piece["template_score"] = shape_score + FIRST_Q_MATCH_AREA_WEIGHT * area_score
        piece["template_shape_score"] = shape_score
        piece["template_area_ratio"] = area_ratio
        piece["template_target_area_ratio"] = template_ratios[template_index]
        piece["template_points"] = len(template["polygon_cm"])
        piece.pop("template_scores", None)


def template_area_match_score(area_ratio, target_ratio):
    area_ratio = max(1e-6, float(area_ratio))
    target_ratio = max(1e-6, float(target_ratio))
    return abs(math.log(area_ratio / target_ratio))


def attach_first_question_targets(pieces):
    target_side = choose_first_question_target_side(pieces)
    final_layout = cached_first_question_target_layout(target_side, use_place_margin=False)
    for piece in pieces:
        template_name = piece.get("template")
        if not template_name or template_name not in final_layout:
            continue

        final_target = final_layout[template_name]
        piece["target_side"] = target_side
        raw_polygon = piece.get("polygon", [])
        current_polygon = piece.get("expanded_polygon", raw_polygon)
        current_angle = polygon_longest_edge_angle(current_polygon)
        target_angle = polygon_longest_edge_angle(final_target["polygon"])
        rotate_deg = estimate_piece_rotation_delta(template_name, current_polygon, final_target["polygon"])
        if rotate_deg is None:
            target_sampled = cached_rotation_target_sample(final_target["polygon"])
            rotate_deg = estimate_polygon_rotation_delta(current_polygon, final_target["polygon"], target_sampled)
            if rotate_deg is None:
                rotate_deg = choose_directed_rotation_180(
                    current_polygon,
                    final_target["polygon"],
                    target_angle - current_angle,
                    target_sampled,
                )
                rotation_method = "edge"
            else:
                rotate_deg = choose_directed_rotation_180(
                    current_polygon,
                    final_target["polygon"],
                    rotate_deg,
                    target_sampled,
                )
                rotation_method = "shape180"
        else:
            rotate_deg = choose_directed_rotation_180(current_polygon, final_target["polygon"], rotate_deg)
            rotation_method = "axisdir"
        target_detected_polygon = place_detected_polygon_on_target(
            current_polygon,
            final_target["polygon"],
            rotate_deg,
        )
        final_detected_polygon = place_detected_polygon_on_target(
            current_polygon,
            final_target["polygon"],
            rotate_deg,
        )
        detected_target_center = polygon_center(target_detected_polygon)
        detected_final_center = polygon_center(final_detected_polygon)
        piece["current_angle"] = round(current_angle, 1)
        piece["target_polygon"] = final_target["polygon"]
        piece["target_center"] = detected_target_center
        piece["target_detected_polygon"] = target_detected_polygon
        piece["final_polygon"] = final_target["polygon"]
        piece["final_center"] = detected_final_center
        piece["final_detected_polygon"] = final_detected_polygon
        piece["target_angle"] = round(target_angle, 1)
        piece["rotate_deg"] = round(rotate_deg, 1)
        piece["rotate_method"] = rotation_method
        piece["move"] = {
            "pick": piece.get("center", [0, 0]),
            "place": detected_target_center,
            "final_place": detected_final_center,
            "rotate_deg": round(rotate_deg, 1),
            "rotate_method": rotation_method,
        }


def choose_first_question_target_side(pieces):
    if FIRST_Q_TARGET_SIDE in ("left", "right"):
        return FIRST_Q_TARGET_SIDE

    centers = [piece.get("center") for piece in pieces if piece.get("center")]
    if not centers:
        return "left"

    mean_x = float(np.mean([center[0] for center in centers]))
    return "left" if mean_x >= WARP_W * 0.5 else "right"


def first_question_target_layout(target_side=None, use_place_margin=False):
    origin_cm = first_question_target_origin_cm(target_side)
    rect_w, rect_h = first_question_target_rect_size_cm()
    target_rect_center_cm = origin_cm + np.array([rect_w * 0.5, rect_h * 0.5], dtype=np.float32)
    layout = {}
    for template in FIRST_Q_TEMPLATES:
        name = template["name"]
        points_cm = orient_first_question_points(np.asarray(template["polygon_cm"], dtype=np.float32))
        points_cm = points_cm + origin_cm.reshape(1, 2)
        points_cm = target_rect_center_cm.reshape(1, 2) + (
            points_cm - target_rect_center_cm.reshape(1, 2)
        ) * float(PIECE_TARGET_SHAPE_EXPAND_SCALE)
        polygon = [cm_to_warp(point).tolist() for point in points_cm]
        center = np.mean(np.asarray(polygon, dtype=np.float32), axis=0)
        if use_place_margin:
            safe_center = first_question_safe_place(center, target_side)
            shift = np.asarray(safe_center, dtype=np.float32) - center
            polygon = (np.asarray(polygon, dtype=np.float32) + shift.reshape(1, 2)).round().astype(np.int32).tolist()
            center = np.asarray(safe_center, dtype=np.float32)
        layout[name] = {
            "polygon": [[int(x), int(y)] for x, y in polygon],
            "center": [int(round(center[0])), int(round(center[1]))],
        }
    return layout


def cached_first_question_target_layout(target_side=None, use_place_margin=False):
    key = (
        target_side or FIRST_Q_TARGET_SIDE,
        bool(use_place_margin),
        FIRST_Q_TARGET_ORIENTATION,
        FIRST_Q_TARGET_SIDE,
        float(FIRST_Q_PLACE_MARGIN_CM),
        float(PIECE_TARGET_SHAPE_EXPAND_SCALE),
    )
    cached = first_question_target_layout_cache.get(key)
    if cached is None:
        cached = first_question_target_layout(target_side, use_place_margin)
        first_question_target_layout_cache[key] = cached
    return cached


def first_question_safe_place(target_center, target_side):
    if FIRST_Q_PLACE_MARGIN_CM <= 0:
        return target_center

    target_origin_cm = first_question_target_origin_cm(target_side)
    rect_w, rect_h = first_question_target_rect_size_cm()
    rect_center_cm = target_origin_cm + np.array([rect_w * 0.5, rect_h * 0.5], dtype=np.float32)
    rect_center = cm_to_warp(rect_center_cm).astype(np.float32)
    center = np.asarray(target_center, dtype=np.float32)
    direction = center - rect_center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return target_center

    margin_px = FIRST_Q_PLACE_MARGIN_CM * (WARP_W - 1) / A4_W_CM
    safe = center + direction / norm * margin_px
    safe[0] = max(0, min(WARP_W - 1, safe[0]))
    safe[1] = max(0, min(WARP_H - 1, safe[1]))
    return np.rint(safe).astype(np.int32).tolist()


def first_question_target_origin_cm(target_side=None):
    target_side = target_side or FIRST_Q_TARGET_SIDE
    if target_side == "auto":
        target_side = "left"
    half_w = A4_W_CM * 0.5
    rect_w, rect_h = first_question_target_rect_size_cm()
    if target_side == "right":
        x0 = half_w + (half_w - rect_w) * 0.5
    else:
        x0 = (half_w - rect_w) * 0.5
    y0 = (A4_H_CM - rect_h) * 0.5
    return np.array([x0, y0], dtype=np.float32)


def first_question_target_rect_size_cm():
    if FIRST_Q_TARGET_ORIENTATION == "portrait":
        return FIRST_Q_RECT_H_CM, FIRST_Q_RECT_W_CM
    return FIRST_Q_RECT_W_CM, FIRST_Q_RECT_H_CM


def orient_first_question_points(points_cm):
    if FIRST_Q_TARGET_ORIENTATION != "portrait":
        return points_cm

    # Rotate the 10x6 cm template clockwise into a 6x10 cm upright target.
    oriented = np.zeros_like(points_cm, dtype=np.float32)
    oriented[:, 0] = FIRST_Q_RECT_H_CM - points_cm[:, 1]
    oriented[:, 1] = points_cm[:, 0]
    return oriented


def cm_to_warp(point_cm):
    point = np.asarray(point_cm, dtype=np.float32)
    x = point[0] * (WARP_W - 1) / A4_W_CM
    y = point[1] * (WARP_H - 1) / A4_H_CM
    return np.rint([x, y]).astype(np.int32)


def expand_polygon_about_center(points, center, scale):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center_array = np.asarray(center, dtype=np.float32).reshape(1, 2)
    expanded = center_array + (points_array - center_array) * float(scale)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, WARP_W - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, WARP_H - 1)
    return np.rint(expanded).astype(np.int32).tolist()


def place_detected_polygon_on_target(points, target_points, rotate_deg):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    target_array = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    source_sampled = resample_closed_polygon(points_array, ROTATION_MATCH_SAMPLE_COUNT)
    target_sampled = resample_closed_polygon(target_array, ROTATION_MATCH_SAMPLE_COUNT)
    if source_sampled is None:
        source_center = np.mean(points_array, axis=0).reshape(1, 2)
    else:
        source_center = np.mean(source_sampled, axis=0).reshape(1, 2)
    if target_sampled is None:
        target_center = np.mean(target_array, axis=0).reshape(1, 2)
    else:
        target_center = np.mean(target_sampled, axis=0).reshape(1, 2)

    centered = points_array - source_center
    rotated = rotate_points_clockwise(centered, rotate_deg)
    placed = rotated + target_center
    placed[:, 0] = np.clip(placed[:, 0], 0, WARP_W - 1)
    placed[:, 1] = np.clip(placed[:, 1], 0, WARP_H - 1)
    return np.rint(placed).astype(np.int32).tolist()


def polygon_center(points):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points_array) == 0:
        return [0, 0]
    centroid = contour_centroid(points_array.astype(np.int32).reshape(-1, 1, 2))
    if centroid is not None:
        return centroid
    center = np.mean(points_array, axis=0)
    return [int(round(float(center[0]))), int(round(float(center[1])))]


def contour_centroid(contour):
    moments = cv2.moments(contour)
    if abs(moments["m00"]) <= 1e-6:
        return None
    return [
        int(round(moments["m10"] / moments["m00"])),
        int(round(moments["m01"] / moments["m00"])),
    ]


def polygon_longest_edge_angle(points):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points_array) < 2:
        return 0.0

    best_length = -1.0
    best_angle = 0.0
    for index in range(len(points_array)):
        start = points_array[index]
        end = points_array[(index + 1) % len(points_array)]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length > best_length:
            best_length = length
            best_angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    return normalize_angle_180(best_angle)


def estimate_piece_rotation_delta(template_name, source_points, target_points):
    if template_name not in ("A", "B", "C"):
        return None

    source_angle = polygon_principal_axis_angle(source_points)
    target_angle = polygon_principal_axis_angle(target_points)
    if source_angle is None or target_angle is None:
        return None
    return target_angle - source_angle


def choose_directed_rotation_180(source_points, target_points, rotation_delta, target_sampled=None):
    source_sampled = resample_closed_polygon(np.asarray(source_points, dtype=np.float32), ROTATION_MATCH_SAMPLE_COUNT)
    if target_sampled is None:
        target_sampled = resample_closed_polygon(np.asarray(target_points, dtype=np.float32), ROTATION_MATCH_SAMPLE_COUNT)
    else:
        target_sampled = np.asarray(target_sampled, dtype=np.float32).reshape(-1, 2)
    if source_sampled is None or target_sampled is None:
        return normalize_angle_180(rotation_delta)

    source_centered = source_sampled - np.mean(source_sampled, axis=0)
    target_centered = target_sampled - np.mean(target_sampled, axis=0)
    base_angle = normalize_angle_180(rotation_delta)
    flipped_angle = normalize_angle_180(rotation_delta + 180.0)
    base_score = score_directed_rotation(source_points, target_points, source_centered, target_centered, base_angle)
    flipped_score = score_directed_rotation(source_points, target_points, source_centered, target_centered, flipped_angle)
    if flipped_score < base_score:
        return flipped_angle
    return base_angle


def choose_directed_axis_rotation(source_points, target_points, axis_delta):
    return choose_directed_rotation_180(source_points, target_points, axis_delta)


def score_directed_rotation(source_points, target_points, source_centered, target_centered, angle):
    boundary_score = score_rotation_by_boundary_distance(source_centered, target_centered, angle)
    feature_score = score_rotation_by_feature_direction(source_points, target_points, angle)
    return boundary_score + feature_score


def score_rotation_by_feature_direction(source_points, target_points, angle):
    source_vector = polygon_feature_vector(source_points)
    target_vector = polygon_feature_vector(target_points)
    if source_vector is None or target_vector is None:
        return 0.0
    rotated_vector = rotate_points_clockwise(np.asarray([source_vector], dtype=np.float32), angle)[0]
    source_norm = float(np.linalg.norm(rotated_vector))
    target_norm = float(np.linalg.norm(target_vector))
    if source_norm <= 1e-6 or target_norm <= 1e-6:
        return 0.0
    rotated_vector = rotated_vector / source_norm
    target_vector = target_vector / target_norm
    return float(np.sum((rotated_vector - target_vector) * (rotated_vector - target_vector))) * 2000.0


def polygon_feature_vector(points):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points_array) < 3:
        return None

    center = np.mean(points_array, axis=0)
    best_index = None
    best_score = None
    for index in range(len(points_array)):
        prev_point = points_array[(index - 1) % len(points_array)]
        point = points_array[index]
        next_point = points_array[(index + 1) % len(points_array)]
        interior = angle_at(prev_point, point, next_point)
        radius = float(np.linalg.norm(point - center))
        score = interior - radius * 0.02
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    if best_index is None:
        return None
    return points_array[best_index] - center


def polygon_principal_axis_angle(points):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points_array) < 3:
        return None

    centered = points_array - np.mean(points_array, axis=0)
    xx = float(np.sum(centered[:, 0] * centered[:, 0]))
    yy = float(np.sum(centered[:, 1] * centered[:, 1]))
    xy = float(np.sum(centered[:, 0] * centered[:, 1]))
    if xx + yy <= 1e-6:
        return None

    return 0.5 * math.degrees(math.atan2(2.0 * xy, xx - yy))


def estimate_polygon_rotation_delta(source_points, target_points, target_sampled=None):
    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    if len(source) < 3 or len(target) < 3:
        return None

    sample_count = ROTATION_MATCH_SAMPLE_COUNT
    source_sampled = resample_closed_polygon(source, sample_count)
    if target_sampled is None:
        target_sampled = resample_closed_polygon(target, sample_count)
    else:
        target_sampled = np.asarray(target_sampled, dtype=np.float32).reshape(-1, 2)
    if source_sampled is None or target_sampled is None:
        return None

    source_centered = source_sampled - np.mean(source_sampled, axis=0)
    if np.linalg.norm(source_centered) <= 1e-6:
        return None
    target_centered_sampled = target_sampled - np.mean(target_sampled, axis=0)
    if np.linalg.norm(target_centered_sampled) <= 1e-6:
        return None

    best_angle = None
    best_score = None
    for angle in edge_alignment_angle_candidates(source, target)[:ROTATION_MATCH_MAX_CANDIDATES]:
        score = score_rotation_by_boundary_distance(source_centered, target_centered_sampled, angle)
        if best_score is None or score < best_score:
            best_score = score
            best_angle = angle

    if len(source) == len(target):
        for candidate in polygon_order_variants(target):
            target_centered = candidate - np.mean(candidate, axis=0)
            if np.linalg.norm(target_centered) <= 1e-6:
                continue

            angle = rotation_angle_between_point_sets(source - np.mean(source, axis=0), target_centered)
            score = score_rotation_by_boundary_distance(source_centered, target_centered_sampled, angle)
            if best_score is None or score < best_score:
                best_score = score
                best_angle = angle

    if best_angle is None:
        return None
    return normalize_angle_180(best_angle)


def edge_alignment_angle_candidates(source, target):
    candidates = []
    for source_a, source_b in polygon_edges(source):
        source_vec = source_b - source_a
        source_len = float(np.linalg.norm(source_vec))
        if source_len <= 1e-6:
            continue
        source_angle = math.atan2(float(source_vec[1]), float(source_vec[0]))
        for target_a, target_b in polygon_edges(target):
            for first, second in ((target_a, target_b), (target_b, target_a)):
                target_vec = second - first
                target_len = float(np.linalg.norm(target_vec))
                if target_len <= 1e-6:
                    continue
                length_ratio = source_len / target_len
                if length_ratio < 0.35 or length_ratio > 2.85:
                    continue
                target_angle = math.atan2(float(target_vec[1]), float(target_vec[0]))
                length_score = abs(math.log(max(1e-6, length_ratio)))
                candidates.append((length_score, math.degrees(target_angle - source_angle)))
    candidates.sort(key=lambda item: item[0])
    return [angle for _score, angle in candidates]


def polygon_edges(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]


def score_rotation_by_boundary_distance(source_centered, target_centered, angle):
    rotated = rotate_points_clockwise(source_centered, angle)
    scale = optimal_point_set_scale(rotated, target_centered)
    rotated = rotated * scale
    forward = mean_nearest_point_distance_sq(rotated, target_centered)
    backward = mean_nearest_point_distance_sq(target_centered, rotated)
    return forward + backward


def mean_nearest_point_distance_sq(points, candidates):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
    diffs = points[:, None, :] - candidates[None, :, :]
    distances = np.sum(diffs * diffs, axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def resample_closed_polygon(points, sample_count):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or sample_count < 3:
        return None

    next_points = np.roll(points, -1, axis=0)
    edge_vectors = next_points - points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    perimeter = float(np.sum(edge_lengths))
    if perimeter <= 1e-6:
        return None

    distances = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    sampled = []
    edge_index = 0
    for distance in distances:
        while edge_index < len(edge_lengths) - 1 and distance >= cumulative[edge_index + 1]:
            edge_index += 1
        edge_length = max(1e-6, float(edge_lengths[edge_index]))
        t = (distance - cumulative[edge_index]) / edge_length
        sampled.append(points[edge_index] + edge_vectors[edge_index] * t)
    return np.asarray(sampled, dtype=np.float32)


def polygon_order_variants(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    variants = []
    for ordered in (points, points[::-1]):
        for shift in range(len(ordered)):
            variants.append(np.roll(ordered, -shift, axis=0))
    return variants


def rotation_angle_between_point_sets(source_centered, target_centered):
    cross_sum = float(np.sum(source_centered[:, 0] * target_centered[:, 1] - source_centered[:, 1] * target_centered[:, 0]))
    dot_sum = float(np.sum(source_centered[:, 0] * target_centered[:, 0] + source_centered[:, 1] * target_centered[:, 1]))
    return math.degrees(math.atan2(cross_sum, dot_sum))


def rotate_points_clockwise(points, angle_deg):
    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return np.asarray(points, dtype=np.float32).dot(matrix.T)


def optimal_point_set_scale(source_points, target_points):
    denominator = float(np.sum(source_points * source_points))
    if denominator <= 1e-6:
        return 1.0
    scale = float(np.sum(source_points * target_points)) / denominator
    if scale <= 1e-6:
        return 1.0
    return scale


def normalize_angle_180(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def build_move_plan(pieces):
    plan = []
    for piece in pieces:
        move = piece.get("move")
        if not move:
            continue
        pick = move["pick"]
        place = move.get("final_place", move["place"])
        plan.append({
            "piece": piece.get("template", "?"),
            "pick": pick,
            "place": move["place"],
            "final_place": place,
            "pick_mech": warp_to_mech_point(pick),
            "place_mech": warp_to_mech_point(place),
            "rotate_deg": move["rotate_deg"],
            "rotate_method": move.get("rotate_method", "?"),
            "target_side": piece.get("target_side", "?"),
        })
    return plan


def aggregate_capture_samples(samples):
    valid_samples = [sample for sample in samples if sample.get("status") and sample.get("pieces")]
    if len(valid_samples) < CAPTURE_MIN_VALID_FRAMES:
        return None, []
    valid_samples = filter_a4_capture_samples(valid_samples)
    if len(valid_samples) < CAPTURE_MIN_VALID_FRAMES:
        return None, []

    grouped = {}
    for sample in valid_samples:
        for piece in sample.get("pieces", []):
            template_name = piece.get("template")
            if template_name:
                grouped.setdefault(template_name, []).append(piece)

    aggregated_pieces = []
    for template_name in ("A", "B", "C", "D"):
        pieces = grouped.get(template_name, [])
        if len(pieces) < CAPTURE_MIN_VALID_FRAMES:
            continue
        filtered = filter_piece_samples(pieces)
        if len(filtered) < CAPTURE_MIN_VALID_FRAMES:
            continue
        aggregated_pieces.append(average_piece_samples(template_name, filtered))

    if len(aggregated_pieces) < min(PIECE_MAX_COUNT, len(FIRST_Q_TEMPLATES)):
        return None, []

    aggregated_pieces.sort(key=lambda piece: (piece["center"][1], piece["center"][0]))
    for piece_id, piece in enumerate(aggregated_pieces):
        piece["id"] = piece_id

    corners = average_a4_corners(valid_samples)
    result = {
        "status": True,
        "stable": True,
        "corners": corners,
        "midline": standard_midline(),
        "pieces_count": len(aggregated_pieces),
        "pieces": aggregated_pieces,
        "move_plan": build_move_plan(aggregated_pieces),
        "capture_frames": len(valid_samples),
    }
    piece_contours = [
        np.asarray(piece["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        for piece in aggregated_pieces
    ]
    return result, piece_contours


def filter_a4_capture_samples(samples):
    corner_sets = [sample.get("corners") for sample in samples if sample.get("corners") is not None]
    if len(corner_sets) < CAPTURE_MIN_VALID_FRAMES:
        return []
    corners = np.asarray(corner_sets, dtype=np.float32).reshape(-1, 4, 2)
    median_corners = np.median(corners, axis=0)
    filtered = []
    for sample, sample_corners in zip(samples, corners):
        corner_error = float(np.max(np.linalg.norm(sample_corners - median_corners, axis=1)))
        if corner_error <= CAPTURE_OUTLIER_A4_CORNER_PX:
            filtered.append(sample)
    return filtered


def average_a4_corners(samples):
    corners = np.asarray([sample.get("corners") for sample in samples], dtype=np.float32).reshape(-1, 4, 2)
    mean = np.mean(corners, axis=0)
    return np.rint(mean).astype(np.int32).tolist()


def filter_piece_samples(pieces):
    centers = np.asarray([piece.get("center", [0, 0]) for piece in pieces], dtype=np.float32)
    rotates = np.asarray([float(piece.get("rotate_deg", 0.0)) for piece in pieces], dtype=np.float32)
    median_center = np.median(centers, axis=0)
    median_rotate = float(np.median(rotates))
    filtered = []
    for piece, center, rotate in zip(pieces, centers, rotates):
        center_error = float(np.linalg.norm(center - median_center))
        rotate_error = abs(normalize_angle_180(float(rotate) - median_rotate))
        if center_error <= CAPTURE_OUTLIER_CENTER_PX and rotate_error <= CAPTURE_OUTLIER_ROTATE_DEG:
            filtered.append(piece)
    return filtered


def average_piece_samples(template_name, pieces):
    center = average_points([piece.get("center", [0, 0]) for piece in pieces])
    target_center = average_points([piece.get("target_center", center) for piece in pieces])
    final_center = average_points([piece.get("final_center", target_center) for piece in pieces])
    rotate_deg = average_angles([float(piece.get("rotate_deg", 0.0)) for piece in pieces])
    polygon = average_polygon_field(pieces, "polygon")
    expanded_polygon = average_polygon_field(pieces, "expanded_polygon") or polygon
    target_detected_polygon = average_polygon_field(pieces, "target_detected_polygon")
    final_detected_polygon = average_polygon_field(pieces, "final_detected_polygon") or target_detected_polygon

    averaged = {
        "id": 0,
        "area": int(round(float(np.mean([piece.get("area", 0) for piece in pieces])))),
        "source_area": int(round(float(np.mean([piece.get("source_area", 0) for piece in pieces])))),
        "center": center,
        "bbox": bbox_from_polygon(polygon),
        "side": "left" if center[0] < WARP_W // 2 else "right",
        "points": len(polygon),
        "polygon": polygon,
        "expanded_polygon": expanded_polygon,
        "template": template_name,
        "target_side": pieces[0].get("target_side", "?"),
        "target_polygon": average_polygon_field(pieces, "target_polygon") or target_detected_polygon or polygon,
        "target_center": target_center,
        "target_detected_polygon": target_detected_polygon,
        "final_polygon": average_polygon_field(pieces, "final_polygon") or final_detected_polygon or polygon,
        "final_center": final_center,
        "final_detected_polygon": final_detected_polygon,
        "rotate_deg": round(rotate_deg, 1),
        "rotate_method": "avg",
        "move": {
            "pick": center,
            "place": target_center,
            "final_place": final_center,
            "rotate_deg": round(rotate_deg, 1),
            "rotate_method": "avg",
        },
    }
    return averaged


def average_points(points):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    mean = np.mean(points_array, axis=0)
    return [int(round(float(mean[0]))), int(round(float(mean[1])))]


def average_polygon_field(pieces, field_name):
    polygons = [piece.get(field_name) for piece in pieces if piece.get(field_name)]
    if not polygons:
        return None
    point_count = len(polygons[0])
    if any(len(polygon) != point_count for polygon in polygons):
        return polygons[0]
    points = np.asarray(polygons, dtype=np.float32)
    mean = np.mean(points, axis=0)
    return np.rint(mean).astype(np.int32).tolist()


def average_angles(angles):
    radians = np.radians(np.asarray(angles, dtype=np.float32))
    sine = float(np.mean(np.sin(radians)))
    cosine = float(np.mean(np.cos(radians)))
    if abs(sine) <= 1e-6 and abs(cosine) <= 1e-6:
        return float(np.mean(angles))
    return normalize_angle_180(math.degrees(math.atan2(sine, cosine)))


def bbox_from_polygon(polygon):
    if not polygon:
        return [0, 0, 0, 0]
    x, y, w, h = cv2.boundingRect(np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2))
    return [int(x), int(y), int(w), int(h)]


def mech_calibration_matrix():
    src = np.float32(
        [
            [0.0, 0.0],
            [WARP_W - 1.0, 0.0],
            [WARP_W - 1.0, WARP_H - 1.0],
            [0.0, WARP_H - 1.0],
        ]
    )
    dst = np.float32(MECH_CALIBRATION_POINTS)
    return cv2.getPerspectiveTransform(src, dst)


def warp_to_mech_point(point):
    if not MECH_COORD_OUTPUT_ENABLED:
        return format_output_mech_point([float(point[0]), float(point[1])])

    matrix = mech_calibration_matrix()
    src = np.float32([[[float(point[0]), float(point[1])]]])
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    return format_output_mech_point([float(dst[0]), float(dst[1])])


def format_output_mech_point(point):
    if MECH_SWAP_XY_FOR_STM32:
        return [float(point[1]), float(point[0])]
    return [float(point[0]), float(point[1])]


def format_mech_value(value):
    if MECH_COORD_DECIMALS <= 0:
        return str(int(round(float(value))))
    return ("%." + str(MECH_COORD_DECIMALS) + "f") % float(value)


def piece_name_to_id(name):
    if name == "A":
        return 0
    if name == "B":
        return 1
    if name == "C":
        return 2
    if name == "D":
        return 3
    return 9


def format_rotate_value(angle_deg):
    return str(int(round(float(angle_deg) * MECH_ROTATE_SCALE)))


def move_plan_records(result):
    if not result.get("status"):
        return []

    move_plan = result.get("move_plan", [])
    if not move_plan:
        return []

    records = []
    for move in sorted(move_plan, key=lambda item: item.get("piece", "?")):
        piece_name = move.get("piece", "?")
        pick_x, pick_y = move.get("pick_mech", move.get("pick", [0, 0]))
        place_x, place_y = move.get("place_mech", move.get("final_place", move.get("place", [0, 0])))
        rotate_deg = float(move.get("rotate_deg", 0.0))
        records.append(
            {
                "name": piece_name,
                "id": piece_name_to_id(piece_name),
                "pick_x": int(round(float(pick_x))),
                "pick_y": int(round(float(pick_y))),
                "place_x": int(round(float(place_x))),
                "place_y": int(round(float(place_y))),
                "rotate_deg": rotate_deg,
                "rotate_x10": int(round(rotate_deg * MECH_ROTATE_SCALE)),
                "rotate_method": move.get("rotate_method", "?"),
            }
        )
    return records


def build_move_packet_text(result):
    records = move_plan_records(result)
    if not records:
        return None

    fields = ["MV", str(len(records))]
    for record in records:
        fields.extend(
            [
                str(record["id"]),
                str(record["pick_x"]),
                str(record["pick_y"]),
                str(record["place_x"]),
                str(record["place_y"]),
                str(record["rotate_x10"]),
            ]
        )
    return " ".join(fields)


def build_move_packet_binary(result):
    records = move_plan_records(result)
    if not records:
        return None

    payload = bytearray()
    payload.append(len(records))
    for record in records:
        payload.extend(
            struct.pack(
                "<Bhhhhh",
                record["id"],
                record["pick_x"],
                record["pick_y"],
                record["place_x"],
                record["place_y"],
                record["rotate_x10"],
            )
        )

    length = len(payload)
    checksum = (length + sum(payload)) & 0xFF
    packet = bytearray([0xAA, 0x55, length])
    packet.extend(payload)
    packet.append(checksum)
    return bytes(packet)


def format_gcode_value(value):
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return ("%.3f" % number).rstrip("0").rstrip(".")


def build_move_packet_gcode(result):
    records = move_plan_records(result)
    if not records:
        return None

    lines = [
        "G21",
        "G90",
    ]
    for record in records:
        lines.append("; piece %s rot=%.1f/%s" % (record["name"], record["rotate_deg"], record["rotate_method"]))
        lines.append("G1 Z%s F%s" % (format_gcode_value(GCODE_PEN_UP_Z), format_gcode_value(GCODE_FEEDRATE)))
        lines.append(
            "G0 X%s Y%s F%s"
            % (
                format_gcode_value(record["pick_x"]),
                format_gcode_value(record["pick_y"]),
                format_gcode_value(GCODE_TRAVEL_FEEDRATE),
            )
        )
        lines.append("G1 Z%s F%s" % (format_gcode_value(GCODE_PEN_DOWN_Z), format_gcode_value(GCODE_FEEDRATE)))
        if GCODE_ROTATE_AXIS:
            lines.append(
                "G1 %s%s F%s"
                % (
                    GCODE_ROTATE_AXIS,
                    format_gcode_value(record["rotate_deg"]),
                    format_gcode_value(GCODE_FEEDRATE),
                )
            )
        lines.append(
            "G1 X%s Y%s F%s"
            % (
                format_gcode_value(record["place_x"]),
                format_gcode_value(record["place_y"]),
                format_gcode_value(GCODE_FEEDRATE),
            )
        )
        lines.append("G1 Z%s F%s" % (format_gcode_value(GCODE_PEN_UP_Z), format_gcode_value(GCODE_FEEDRATE)))
    return "\n".join(lines)


def build_move_packet_log(result):
    records = move_plan_records(result)
    if not records:
        return None

    parts = ["MOVES %d" % len(records)]
    for record in records:
        parts.append(
            "%s pick=(%d,%d) place=(%d,%d) rot=%.1f/%s"
            % (
                record["name"],
                record["pick_x"],
                record["pick_y"],
                record["place_x"],
                record["place_y"],
                record["rotate_deg"],
                record["rotate_method"],
            )
        )
    return " | ".join(parts)


def send_text(serial_obj, line):
    print(line)
    if serial_obj is None:
        return

    payload = (line + "\n").encode("utf-8")
    try:
        serial_obj.write(payload)
    except Exception as err:
        print("UART WRITE FAIL %s" % err)


def send_binary(serial_obj, packet, log_line=None):
    if log_line:
        print(log_line)
    else:
        print("TX %d bytes" % len(packet))
    if serial_obj is None:
        return

    try:
        serial_obj.write(packet)
    except Exception as err:
        print("UART WRITE FAIL %s" % err)


def send_move_packet(result, serial_obj):
    if SERIAL_OUTPUT_FORMAT == "binary":
        packet = build_move_packet_binary(result)
        if packet:
            send_binary(serial_obj, packet, build_move_packet_log(result))
        return

    if SERIAL_OUTPUT_FORMAT == "gcode":
        packet = build_move_packet_gcode(result)
    else:
        packet = build_move_packet_text(result)
    if packet:
        send_text(serial_obj, packet)


def print_move_packet(result):
    if SERIAL_OUTPUT_FORMAT == "binary":
        log_line = build_move_packet_log(result)
    elif SERIAL_OUTPUT_FORMAT == "gcode":
        log_line = build_move_packet_log(result)
    else:
        log_line = build_move_packet_text(result)

    if log_line:
        print(log_line)


def get_template_shape_score(piece, template_index):
    for score in piece.get("template_scores", []):
        if score["index"] == template_index:
            return float(score["shape_score"])
    return 999.0


def first_question_template_area_ratios():
    areas = [polygon_area_cm(template["polygon_cm"]) for template in FIRST_Q_TEMPLATES]
    total = max(1e-6, sum(areas))
    return [area / total for area in areas]


def cached_first_question_template_area_ratios():
    global first_question_area_ratios_cache
    if first_question_area_ratios_cache is None:
        first_question_area_ratios_cache = first_question_template_area_ratios()
    return first_question_area_ratios_cache


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


def cached_template_contour_for_match(template_index):
    cached = first_question_template_contour_cache.get(template_index)
    if cached is None:
        cached = template_contour_for_match(FIRST_Q_TEMPLATES[template_index]["polygon_cm"])
        first_question_template_contour_cache[template_index] = cached
    return cached


def cached_rotation_target_sample(target_points):
    key = (
        ROTATION_MATCH_SAMPLE_COUNT,
        tuple((int(point[0]), int(point[1])) for point in target_points),
    )
    cached = rotation_target_sample_cache.get(key)
    if cached is None:
        cached = resample_closed_polygon(np.asarray(target_points, dtype=np.float32), ROTATION_MATCH_SAMPLE_COUNT)
        rotation_target_sample_cache[key] = cached
    return cached


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
            best = approx
            if 3 <= len(approx) <= PIECE_MAX_POINTS:
                break
        epsilon_ratio += PIECE_APPROX_EPSILON_STEP

    if FIRST_QUESTION_MODE and PIECE_REFINE_CORNERS_BY_LINES and best is not None and 3 <= len(best) <= PIECE_MAX_POINTS:
        refined = refine_polygon_corners_by_lines(contour, best)
        if refined is not None:
            best = refined
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


def draw_first_question_targets(frame, pieces):
    if not FIRST_QUESTION_MODE:
        return
    if not pieces:
        return

    target_side = None
    for piece in pieces or []:
        if piece.get("target_side"):
            target_side = piece["target_side"]
            break
    if target_side is None:
        target_side = choose_first_question_target_side(pieces or [])
    layout = cached_first_question_target_layout(target_side, use_place_margin=False)
    layout_points = []
    for target in layout.values():
        layout_points.extend(target["polygon"])
    if layout_points:
        x, y, w, h = cv2.boundingRect(np.asarray(layout_points, dtype=np.int32).reshape(-1, 1, 2))
        cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 1)

    for piece in pieces or []:
        if "target_center" not in piece:
            continue
        if piece.get("target_detected_polygon"):
            detected_polygon = np.asarray(piece["target_detected_polygon"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(frame, [detected_polygon], -1, CYAN, 2)
        tx, ty = piece["target_center"]
        cv2.circle(frame, (int(tx), int(ty)), 3, CYAN, -1)
        if piece.get("template"):
            cv2.putText(frame, piece["template"], (int(tx) + 4, max(12, int(ty) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)


def draw_piece_debug_info(frame, result, fps, piece_debug):
    draw_text_bg(frame, 8, 24, "A4 OK  FPS:%d" % fps, GREEN, 0.7)
    if PIECE_MASK_METHOD == "hsv":
        mask_info = "mode:%d HSV H:%d-%d thr:%d" % (
            DISPLAY_MODE,
            PIECE_GREEN_H_LOW,
            PIECE_GREEN_H_HIGH,
            int(PIECE_HSV_DIFF_THRESHOLD),
        )
    elif PIECE_MASK_METHOD == "black":
        bg_color = piece_debug.get("bg_color")
        threshold = int(bg_color[0]) if bg_color is not None else int(PIECE_BLACK_V_MIN_THRESHOLD)
        bg_v = int(bg_color[1]) if bg_color is not None else 0
        mask_info = "mode:%d BLACK V>%d bg:%d" % (DISPLAY_MODE, threshold, bg_v)
    else:
        mask_info = "mode:%d LAB thr:%d" % (DISPLAY_MODE, int(PIECE_BG_DIFF_THRESHOLD))
    draw_text_bg(frame, 8, 46, mask_info, WHITE, 0.5)
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
    if PIECE_MASK_METHOD == "hsv":
        detail = "S:%d-%d V:%d-%d blur:%dx%d med:%d" % (
            PIECE_GREEN_S_LOW,
            PIECE_GREEN_S_HIGH,
            PIECE_GREEN_V_LOW,
            PIECE_GREEN_V_HIGH,
            PIECE_HSV_DIFF_BLUR_KERNEL[0],
            PIECE_HSV_DIFF_BLUR_KERNEL[1],
            PIECE_MASK_MEDIAN_KERNEL,
        )
    elif PIECE_MASK_METHOD == "black":
        detail = "min:%d off:%d blur:%dx%d med:%d" % (
            PIECE_BLACK_V_MIN_THRESHOLD,
            PIECE_BLACK_V_OFFSET,
            PIECE_BLACK_BLUR_KERNEL[0],
            PIECE_BLACK_BLUR_KERNEL[1],
            PIECE_MASK_MEDIAN_KERNEL,
        )
    else:
        detail = "lw:%.2f" % PIECE_L_DIFF_WEIGHT
    draw_text_bg(frame, 8, 106, "cnt_mask:%s %s" % (piece_debug.get("contour_mask", "?"), detail), WHITE, 0.5)


def gray_to_bgr(gray):
    if gray is None:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def distance_map_to_view(distance_map):
    if distance_map is None:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    if PIECE_MASK_METHOD == "hsv":
        max_value = max(1.0, float(np.percentile(distance_map, 98)))
        view_gray = np.clip(distance_map * (255.0 / max_value), 0, 255).astype(np.uint8)
    elif PIECE_MASK_METHOD == "black":
        view_gray = np.clip(distance_map, 0, 255).astype(np.uint8)
    else:
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
        if PIECE_SPLIT_TOUCHING_ENABLED:
            key_name = "split_mask_original" if DEBUG_SHOW_ORIGINAL_PROCESS else "split_mask"
        else:
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
            draw_first_question_targets(view, pieces or [])
            draw_pieces(view, pieces or [], piece_contours or [])
            draw_warp_result(view, result, fps)
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if DISPLAY_MODE == 1 and result.get("status"):
        view = a4_warp.copy() if a4_warp is not None else warp_a4(frame, result["corners"])[0]
        draw_first_question_targets(view, pieces or [])
        draw_pieces(view, pieces or [], piece_contours or [])
        draw_warp_result(view, result, fps)
        return view

    view = frame.copy()
    if DISPLAY_MODE == 1:
        draw_warp_result(view, result, fps)
    else:
        draw_original_result(view, result, fps)
    return view


def display_requires_a4_warp(result):
    if not result.get("status"):
        return False
    if DISPLAY_MODE == 1:
        return True
    return DISPLAY_MODE in (5, 6) and not DEBUG_SHOW_ORIGINAL_PROCESS


def on_key(key_id, state):
    if ENABLE_KEY_EXIT and state == key.State.KEY_LONG_PRESSED:
        app.set_exit_flag(True)
        return

    if SEND_SERIAL_ON_KEY_PRESS and is_key_send_state(state):
        global capture_requested
        capture_requested = True


def is_key_send_state(state):
    for state_name in ("KEY_PRESSED", "KEY_SHORT_PRESSED"):
        if hasattr(key.State, state_name) and state == getattr(key.State, state_name):
            return True
    return False


def create_camera():
    try:
        return camera.Camera(FRAME_WIDTH, FRAME_HEIGHT, image.Format.FMT_BGR888)
    except Exception:
        return camera.Camera(FRAME_WIDTH, FRAME_HEIGHT)


def create_serial():
    if not SERIAL_OUTPUT_ENABLED or uart is None:
        return None

    try:
        return uart.UART(SERIAL_PORT, SERIAL_BAUDRATE)
    except Exception as err:
        print("UART OPEN FAIL %s" % err)
        return None


def main():
    global capture_requested

    cam = create_camera()
    try:
        cam.skip_frames(30)
    except Exception:
        pass

    disp = display.Display()
    detector = A4Detector()
    stabilizer = A4ResultStabilizer(MAX_LOST_FRAMES)
    key_obj = key.Key(on_key) if ENABLE_KEY_EXIT else None
    serial_obj = create_serial()
    last_print_ms = time.ticks_ms()
    last_fps_ms = last_print_ms
    frame_count = 0
    fps = 0
    perf_accum = {
        "image2cv": 0,
        "a4": 0,
        "matrix": 0,
        "pieces": 0,
        "display": 0,
        "total": 0,
    }
    capture_remaining = 0
    capture_samples = []
    held_result = None
    held_piece_contours = []
    held_matrix = None
    held_until_ms = 0
    capture_wait_logged = False

    while not app.need_exit():
        loop_start_ms = time.ticks_ms()
        maix_img = cam.read()
        image2cv_start_ms = time.ticks_ms()
        frame = image.image2cv(maix_img, ensure_bgr=True, copy=True)
        image2cv_end_ms = time.ticks_ms()

        a4_start_ms = time.ticks_ms()
        raw_result = detector.detect(frame)
        a4_end_ms = time.ticks_ms()
        result = stabilizer.update(raw_result)
        a4_warp = None
        matrix = None
        pieces = []
        piece_contours = []
        piece_debug = make_empty_piece_debug()
        output_result = result
        if result.get("status"):
            matrix_start_ms = time.ticks_ms()
            matrix, _ = a4_perspective_matrices(result["corners"])
            matrix_end_ms = time.ticks_ms()
            pieces_start_ms = time.ticks_ms()
            now_for_capture = time.ticks_ms()
            if capture_requested:
                capture_requested = False
                capture_remaining = CAPTURE_FRAME_COUNT
                capture_samples = []
                capture_wait_logged = False
                print("CAPTURE START frames=%d" % CAPTURE_FRAME_COUNT)

            if capture_remaining > 0:
                pieces, piece_contours, piece_debug = detect_pieces(frame, result["corners"], matrix)
                capture_result = result.copy()
                capture_result["pieces_count"] = len(pieces)
                capture_result["pieces"] = pieces
                if FIRST_QUESTION_MODE:
                    capture_result["move_plan"] = build_move_plan(pieces)
                capture_samples.append(capture_result)
                capture_remaining -= 1
                output_result = capture_result
                if capture_remaining <= 0:
                    aggregated_result, aggregated_contours = aggregate_capture_samples(capture_samples)
                    if aggregated_result is not None:
                        held_result = aggregated_result
                        held_piece_contours = aggregated_contours
                        held_matrix, _ = a4_perspective_matrices(held_result["corners"])
                        held_until_ms = now_for_capture + CAPTURE_HOLD_MS
                        output_result = held_result
                        pieces = held_result.get("pieces", [])
                        piece_contours = held_piece_contours
                        print("CAPTURE OK valid=%d pieces=%d" % (held_result.get("capture_frames", 0), len(pieces)))
                        send_move_packet(held_result, serial_obj)
                    else:
                        held_result = None
                        held_piece_contours = []
                        held_matrix = None
                        held_until_ms = 0
                        print("CAPTURE FAIL valid=%d" % len(capture_samples))
            elif held_result is not None and now_for_capture <= held_until_ms:
                output_result = held_result
                pieces = held_result.get("pieces", [])
                piece_contours = held_piece_contours
            elif held_result is not None and now_for_capture > held_until_ms:
                held_result = None
                held_piece_contours = []
                held_matrix = None
                held_until_ms = 0
            pieces_end_ms = time.ticks_ms()
        else:
            now_for_capture = time.ticks_ms()
            if capture_requested and not capture_wait_logged:
                print("CAPTURE WAIT A4")
                capture_wait_logged = True
            if held_result is not None and now_for_capture <= held_until_ms:
                output_result = held_result
                pieces = held_result.get("pieces", [])
                piece_contours = held_piece_contours
            elif held_result is not None and now_for_capture > held_until_ms:
                held_result = None
                held_piece_contours = []
                held_matrix = None
                held_until_ms = 0
            matrix_start_ms = matrix_end_ms = time.ticks_ms()
            pieces_start_ms = pieces_end_ms = matrix_end_ms

        frame_count += 1
        display_start_ms = time.ticks_ms()
        display_result = result
        display_matrix = matrix
        if output_result is held_result and held_matrix is not None:
            display_result = held_result
            display_matrix = held_matrix
        elif output_result is not result and result.get("status"):
            display_result = result.copy()
            display_result["pieces_count"] = output_result.get("pieces_count", 0)
            display_result["pieces"] = output_result.get("pieces", [])
            display_result["move_plan"] = output_result.get("move_plan", [])
        if display_matrix is not None and display_requires_a4_warp(display_result):
            a4_warp = cv2.warpPerspective(frame, display_matrix, (WARP_W, WARP_H))
        view = make_display_view(frame, display_result, fps, a4_warp, pieces, piece_contours, piece_debug)
        disp.show(image.cv2image(view, bgr=True, copy=True))
        display_end_ms = time.ticks_ms()

        if PERF_PROFILE:
            perf_accum["image2cv"] += image2cv_end_ms - image2cv_start_ms
            perf_accum["a4"] += a4_end_ms - a4_start_ms
            perf_accum["matrix"] += matrix_end_ms - matrix_start_ms
            perf_accum["pieces"] += pieces_end_ms - pieces_start_ms
            perf_accum["display"] += display_end_ms - display_start_ms
            perf_accum["total"] += display_end_ms - loop_start_ms

        now = time.ticks_ms()
        if now - last_fps_ms >= 1000:
            fps = frame_count
            if PERF_PROFILE and frame_count > 0:
                print(
                    "PERF n=%d image2cv=%.1f a4=%.1f matrix=%.1f pieces=%.1f display=%.1f total=%.1f ms"
                    % (
                        frame_count,
                        perf_accum["image2cv"] / frame_count,
                        perf_accum["a4"] / frame_count,
                        perf_accum["matrix"] / frame_count,
                        perf_accum["pieces"] / frame_count,
                        perf_accum["display"] / frame_count,
                        perf_accum["total"] / frame_count,
                    )
                )
                for key_name in perf_accum:
                    perf_accum[key_name] = 0
            frame_count = 0
            last_fps_ms = now

        if now - last_print_ms >= PRINT_INTERVAL_MS:
            if PRINT_MOVE_ONLY:
                print_move_packet(output_result)
            else:
                print(output_result)
            last_print_ms = now

    if key_obj:
        del key_obj


if __name__ == "__main__":
    main()
