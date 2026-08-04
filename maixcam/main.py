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
import heapq
import gc


# =========================
# Config: MaixCAM2 OpenCV A4
# =========================
# 主配置区：负责摄像头采集、A4 纸透视矫正、碎片识别、第一/第二问拼接求解、
# 以及最后输出给 STM32/机械端的坐标。调参数时优先看这一段。

# 摄像头输入分辨率，单位像素；分辨率越高识别越细，但 MaixCAM 处理会更慢。
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# A4 纸透视矫正后的标准画布尺寸，单位像素。
# 程序检测到 A4 四角后，会把纸面拉正到 WARP_W x WARP_H，后续碎片坐标都基于这个坐标系。
WARP_W = 594
WARP_H = 420

# A4 纸真实尺寸。cm 用于题目尺寸换算，mm 用于机械坐标和串口输出。
A4_W_CM = 29.7
A4_H_CM = 21.0
A4_W_MM = A4_W_CM * 10.0
A4_H_MM = A4_H_CM * 10.0

# A4 外框边缘检测预处理参数。
GAUSSIAN_KERNEL = (5, 5)       # 高斯模糊核，先降噪，避免 Canny 检出太多杂边。
CANNY_LOW = 50                 # Canny 低阈值，影响弱边缘保留程度。
CANNY_HIGH = 150               # Canny 高阈值，影响强边缘判定。
MORPH_CLOSE_KERNEL = (5, 5)    # 闭运算核，用来连接断开的 A4 外框边缘。
MORPH_CLOSE_ITERATIONS = 1     # 闭运算次数，过大会让相邻轮廓粘在一起。

# A4 候选轮廓筛选参数。
APPROX_EPSILON_RATIO = 0.025   # 多边形拟合精度比例，用于把 A4 轮廓近似成四边形。
MIN_AREA_RATIO = 0.07          # A4 候选最小面积占整帧比例，过滤小噪声。
MAX_AREA_RATIO = 0.65          # A4 候选最大面积占整帧比例，过滤异常大区域。

# 绿色比赛纸优先检测；白色碎片在绿色区域中只是孔洞，不会改变纸张外轮廓。
A4_GREEN_DETECTION_ENABLED = True
A4_GREEN_MIN_AREA_RATIO = 0.15          # 绿色纸最小画面占比。
A4_GREEN_MAX_AREA_RATIO = 0.92          # 允许纸张靠近镜头，占据大部分画面。
A4_GREEN_MIN_COVERAGE_RATIO = 0.48      # 四边形内至少有多少比例仍是绿色，容纳白色碎片和阴影。
A4_GREEN_MEDIAN_KERNEL = 5              # 仅平滑 A4 颜色掩膜，不参与碎片轮廓提取。
A4_GREEN_CLOSE_KERNEL = (9, 9)          # 连接绿色纸外边缘的小断点。
A4_GREEN_APPROX_EPSILONS = (0.008, 0.012, 0.018, 0.025, 0.035, 0.050)

# 防止把中间黑线分出的半张纸当成 A4。
# 如果同一帧里存在更大的 A4 外轮廓，即使它没有拟合成四边形，小候选也会被过滤。
CANDIDATE_MIN_RELATIVE_AREA = 0.65

# A4 外框允许连续丢失的帧数；短暂遮挡/抖动时不立刻清空检测状态。
MAX_LOST_FRAMES = 5
A4_STABILIZER_ALPHA = 0.30              # 同一检测来源的小抖动低通系数，新帧占比。
A4_STABILIZER_MAX_CORNER_JUMP_PX = 35.0 # 超过该角点跳变不做平均，避免错误四边形拖偏结果。
A4_EDGE_SWITCH_CONFIRM_FRAMES = 3       # 已有绿色结果时，接近的 edge 回退需连续确认帧数。
A4_EDGE_SWITCH_TOLERANCE_PX = 18.0      # 连续 edge 候选之间允许的最大角点变化。

# 题目模式和碎片识别总开关。
PIECE_DETECTION_ENABLED = True     # 是否启用碎片检测；False 时只检测/显示 A4 外框。
QUESTION_MODE = 2                  # 当前题号：1 使用第一问模板匹配，2 使用第二问通用拼接。
FIRST_QUESTION_MODE = QUESTION_MODE == 1
SECOND_QUESTION_MODE = QUESTION_MODE == 2
PIECE_MASK_METHOD = "hsv"        # 碎片分割方式；hsv 表示按绿色背景和 HSV 色差提取碎片。
PIECE_PROCESS_A4_ROI = True        # 是否只在透视后的 A4 区域内处理碎片，减少画面外干扰。

# 背景采样和 Lab 色差参数。
PIECE_BG_BORDER_SAMPLE = 24        # 从 A4 边缘向内采样背景的宽度，单位像素。
PIECE_BG_DIFF_THRESHOLD = 35.0     # 与背景色差超过该阈值才认为可能是碎片。
PIECE_L_DIFF_WEIGHT = 0.25         # Lab 亮度 L 通道权重，较低可减小阴影/光照影响。
PIECE_A_DIFF_WEIGHT = 1.0          # Lab a 通道权重，主要描述红绿方向差异。
PIECE_B_DIFF_WEIGHT = 1.0          # Lab b 通道权重，主要描述黄蓝方向差异。

# HSV 绿色范围；用于绿色碎片/纸面时的辅助分割。
PIECE_GREEN_H_LOW = 51             # H 色相下限。
PIECE_GREEN_H_HIGH = 91            # H 色相上限。
PIECE_GREEN_S_LOW = 52             # S 饱和度下限，太低说明颜色偏灰。
PIECE_GREEN_S_HIGH = 255           # S 饱和度上限。
PIECE_GREEN_V_LOW = 75             # V 亮度下限。
PIECE_GREEN_V_HIGH = 255           # V 亮度上限。
# 绿色模式只把明确满足 H/S 条件的像素当背景；低饱和白边不会再因绿色反光被吃掉。
PIECE_HSV_MEDIAN_ENABLED = True    # 是否对绿色模式最终二值 mask 做 3x3 中值滤波；尖角仍被削时可关闭。
PIECE_DEBUG_OVERLAY_RAW_CONTOUR = False # 模式 1 同屏显示红色原始轮廓和蓝色最终多边形。

# 黑色背景分割参数；现场黑底时，用 V 通道亮度把碎片从背景中提出来。
# 这里的阈值比第一问参考文件更低，是为了保留阴影区或反光较弱的碎片角点。
PIECE_BLACK_V_MIN_THRESHOLD = 90   # V 通道绝对亮度阈值，亮于它才可能是碎片。
PIECE_BLACK_V_OFFSET = 55          # 相对背景亮度偏移，用于适应不同灯光环境。
PIECE_BLACK_BLUR_KERNEL = (3, 3)   # 黑底分割前的模糊核。

# 碎片 mask 清理参数。
PIECE_MASK_MEDIAN_KERNEL = 3       # 中值滤波核，去除椒盐噪声。
PIECE_MASK_OPEN_KERNEL = (3, 3)    # 开运算核，用于去小白点。
PIECE_MASK_CLOSE_KERNEL = (3, 3)   # 闭运算核，用于补小断口。
# OPEN 会直接削掉凸出的尖角。背景差分已经比较干净，默认关闭，只保留小核 CLOSE 补断口。
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
PIECE_MAX_POINTS = 5                   # 单个碎片最多角点数；题目碎片一般是 3~5 点。
PIECE_MAX_COUNT = 4                    # 最多识别/求解 4 块碎片。
PIECE_MIN_AREA_RATIO = 0.002           # 碎片最小面积占 A4 区域比例，过滤小噪声。
PIECE_MAX_AREA_RATIO = 0.30            # 碎片最大面积占 A4 区域比例，过滤误检大块。
PIECE_BORDER_MARGIN = 8                # 过滤贴近整帧边框的干扰轮廓，单位像素。
PIECE_ALLOW_A4_EDGE_TOUCH = True        # 允许碎片贴 A4 边缘；否则角在纸边时容易被误过滤/削掉。
PIECE_A4_MASK_INNER_SCALE = 1.0         # A4 mask 内缩比例；1.0 表示不内缩，避免贴边碎片被裁角。
PIECE_FRAME_MASK_MARGIN = 0            # A4 mask 腐蚀边距；第二问贴边时设 0，避免削掉边缘角。
PIECE_MIN_BBOX_SIDE = 8                # 外接框最小边长，过滤极小轮廓。
PIECE_MAX_ASPECT_RATIO = 8.0           # 外接框最大长宽比，过滤细长线状噪声。
PIECE_TARGET_SHAPE_EXPAND_SCALE = 1.00 # 保持实物真实尺寸；第二问安全间距改用固定毫米值。
SECOND_Q_TARGET_GAP_MM = 5.0             # 拼接完成后相邻碎片的计划安全间距，固定毫米值。
SECOND_Q_CLEARANCE_CONTACT_PARALLEL_MIN = 0.94 # 自动补充接缝时，两边方向平行度下限。
SECOND_Q_CLEARANCE_CONTACT_DISTANCE_PX = 12.0 # 自动补充接缝时，两条支撑线最大距离。
SECOND_Q_CLEARANCE_CONTACT_OVERLAP_PX = 10.0 # 自动补充接缝时，沿边方向最小重合长度。

# 第一问目标矩形和模板参数，单位厘米。
FIRST_Q_RECT_W_CM = 10.0               # 第一问目标矩形宽度。
FIRST_Q_RECT_H_CM = 6.0                # 第一问目标矩形高度。
FIRST_Q_TARGET_SIDE = "auto"           # 目标区域放置侧；auto 表示程序自动选择。
FIRST_Q_TARGET_ORIENTATION = "portrait"# 目标布局方向；portrait 表示竖向显示/输出。
FIRST_Q_PLACE_MARGIN_CM = 0.0          # 目标放置外边距，单位厘米。

# 第一问官方切割模板关键点，坐标原点是 10 cm x 6 cm 矩形左上角。
FIRST_Q_DIAG_A = [2.0, 0.0]            # 模板中的 A 点，位于上边。
FIRST_Q_DIAG_P = [3.6, 1.2]            # 模板中的内部连接点 P。
FIRST_Q_DIAG_Q = [7.6, 4.2]            # 模板中的内部连接点 Q。
FIRST_Q_MATCH_SHAPE_WEIGHT = 1.0       # 模板匹配中形状相似度权重。
FIRST_Q_MATCH_AREA_WEIGHT = 4.0        # 模板匹配中面积比例权重。
FIRST_Q_MATCH_POINT_WEIGHT = 0.08      # 模板匹配中角点距离权重。
ROTATION_MATCH_SAMPLE_COUNT = 32       # 旋转匹配时采样多少个角度。
ROTATION_MATCH_MAX_CANDIDATES = 12     # 每块碎片最多保留多少个旋转候选。

# 第一问四块标准模板，单位厘米；A/B/C/D 是目标矩形被切开后的四块标准形状。
FIRST_Q_TEMPLATES = [
    {"name": "A", "polygon_cm": [[0.0, 0.0], FIRST_Q_DIAG_A, FIRST_Q_DIAG_P, [0.0, 2.0]]},
    {"name": "B", "polygon_cm": [[0.0, 2.0], FIRST_Q_DIAG_P, FIRST_Q_DIAG_Q, [0.0, 3.0]]},
    {"name": "C", "polygon_cm": [[0.0, 3.0], FIRST_Q_DIAG_Q, [10.0, 6.0], [0.0, 6.0]]},
    {"name": "D", "polygon_cm": [FIRST_Q_DIAG_A, [10.0, 0.0], [10.0, 6.0], FIRST_Q_DIAG_Q, FIRST_Q_DIAG_P]},
]

# 第二问目标矩形尺寸范围，单位厘米；求解后会尽量拼成满足该尺寸范围的矩形。
SECOND_Q_RECT_MIN_W_CM = 9.0           # 允许的最小宽度。
SECOND_Q_RECT_MIN_H_CM = 5.0           # 允许的最小高度。
SECOND_Q_RECT_MAX_W_CM = 12.0          # 允许的最大宽度。
SECOND_Q_RECT_MAX_H_CM = 9.0           # 允许的最大高度。
SECOND_Q_TARGET_SIDE = "auto"          # 第二问目标放置侧；auto 根据当前碎片位置选左/右。
SECOND_Q_CENTERLINE_GAP_CM = 1.0       # 目标区和 A4 中线之间预留间隙，单位厘米。

# 第二问边匹配候选参数。
SECOND_Q_MATCH_REL_TOLERANCE = 0.12    # 两条整边长度相差比例小于它才认为可能完整匹配。
# 窗口放宽以覆盖深 T(如 0.9 占比);惩罚保持平坦,深 T 同样是合法拓扑。
SECOND_Q_PARTIAL_RATIO_MIN = 0.12      # T 形/局部边匹配的最小占比。
SECOND_Q_PARTIAL_RATIO_MAX = 0.92      # T 形/局部边匹配的最大占比。
SECOND_Q_COMPOSITE_EDGE_MAX_SPAN = 3   # 允许把连续几条近共线边合成一条复合边。
SECOND_Q_COMPOSITE_EDGE_MIN_CHORD_RATIO = 0.96 # 复合边弦长/路径长下限，越接近 1 越要求共线。
SECOND_Q_COMPOSITE_EDGE_MAX_DEVIATION_PX = 12.0# 复合边中间点偏离弦线的最大像素距离。

# 第二问拼接评分和尺寸容差参数。
SECOND_Q_SOLVE_PADDING_PX = 10         # 光栅化评分时给拼接结果周围留的空白边。
SECOND_Q_DIMENSION_TOLERANCE_CM = 0.8  # 目标矩形尺寸容差，单位厘米。
SECOND_Q_MIN_RECT_FILL_RATIO = 0.45    # 最低矩形填充率，太低说明拼得很散。
SECOND_Q_GOOD_RECT_FILL_RATIO = 0.88   # 较好填充率参考值，用于评分/调试判断。
SECOND_Q_MAX_OVERLAP_RATIO = 0.12      # 拼接允许的最大重叠比例。
SECOND_Q_GOOD_OVERLAP_RATIO = 0.02     # 较好重叠比例参考值。
SECOND_Q_RECT_APPROX_EPSILON_RATIO = 0.025 # 矩形轮廓近似精度比例。

# 第二问组合枚举规模控制；这些参数直接影响速度和内存。
SECOND_Q_FULL_MATCHES_PER_PAIR = 8     # 每对碎片保留多少个完整边匹配候选。
# partial 不设过小截断:局部特征无法可靠地区分真假 T 匹配,真值靠评分函数裁决。
SECOND_Q_PARTIAL_MATCHES_PER_PAIR = 22 # 每对碎片保留多少个局部/T 边匹配候选。
# 组合惩罚预算剪枝:超过该预算的组合不可能进入最终短名单。
SECOND_Q_COMBO_PENALTY_BUDGET = 0.75
# 惩罚升序截断:长度/角度证据最好的组合先进评分,控制最坏耗时。
SECOND_Q_MAX_SCORED_COMBOS = 2500      # 通用回退保留更多组合，避免 800 个候选截断正确答案。
SECOND_Q_MAX_SOLVE_COMBOS = 2500       # 预留的最大求解组合数上限。
SECOND_Q_MAX_MATCHING_COMBO_CHECKS = 20000 # 预留的匹配组合检查上限。
SECOND_Q_SOLVE_TIME_LIMIT_MS = 60000   # 视觉求解总硬时间预算；给机械抓放预留约 20 秒以上。
SECOND_Q_AUTO_GEOMETRY_FINALISTS = 384 # 纯几何预筛后进入小图粗筛的候选数。
SECOND_Q_AUTO_COARSE_MASK_SCALE = 0.16 # auto 全量粗筛掩膜比例；只用于排序，不作为最终验收结果。
SECOND_Q_AUTO_FINALISTS = 96           # 粗筛后进入原完整评分的候选数，兼顾正确率和板端耗时。
# 新拼接算法以光栅化掩膜评分为准(同参考实现);凸包近似评分对凹片/五边形会误判。
SECOND_Q_USE_FAST_GEOMETRY_SCORE = False

# 第二问切割拓扑和调试开关。
# Try a known topology first: auto/common/t_junction/corner/concave/equal_rectangles/branched_spine.
# Use "all_reference" to scan the reference simulator's named modes.
SECOND_Q_CUT_MODE = "branched_spine"   # 当前优先使用的参考拓扑名称，实际求解阶段会走 auto 枚举。
SECOND_Q_REFERENCE_MATCHING = True     # 是否启用参考拓扑风格的边匹配策略。
SECOND_Q_SOLVER_DEBUG_VERSION = "q2topology-v56-clearance-contacts" # 日志版本，确认 MaixVision 未覆盖新文件。
# 第二问遇到第一问官方切法时先按官方邻接拓扑跑第二问算法,失败再回全量通用搜索。
SECOND_Q_FIRST_TEMPLATE_DETECT_ENABLED = True
SECOND_Q_FIRST_TEMPLATE_TOPOLOGY_FIRST = True
SECOND_Q_FIRST_TEMPLATE_MAX_COST = 1.60
SECOND_Q_FIRST_TEMPLATE_MAX_SHAPE_SCORE = 0.42
SECOND_Q_FIRST_TEMPLATE_AREA_WEIGHT = 0.80       # 打乱摆放时面积/阴影波动较大,这里只弱化面积约束。
SECOND_Q_FIRST_TEMPLATE_POINT_WEIGHT = 0.04      # 角点数量只作弱参考,避免拟合多/少一个点就漏检。
SECOND_Q_FIRST_TEMPLATE_CANDIDATES_PER_PAIR = 16  # 放宽每对接缝候选，容纳裁剪和轮廓抖动。
SECOND_Q_FIRST_TEMPLATE_MAX_SCORED_COMBOS = 900   # 官方拓扑范围小，可保留更完整的组合。
SECOND_Q_FIRST_TEMPLATE_RING_CANDIDATES_PER_PAIR = 8  # 外圈快路按模板接缝长度排序后每对保留数量。
SECOND_Q_FIRST_TEMPLATE_RING_MAX_SCORED_COMBOS = 160 # 外圈四边闭环优先评分上限。
SECOND_Q_FIRST_TEMPLATE_GUIDED_MAPS_PER_PIECE = 6 # 每块保留多套模板顶点映射，抵抗轮廓抖动和多余角点。
SECOND_Q_FIRST_TEMPLATE_GUIDED_MAX_SETS = 24      # 四块映射组合后最多尝试的官方引导拼法数量。
SECOND_Q_FIRST_TEMPLATE_CONFIDENT_MAX_COST = 0.85 # 只有高置信模板命中才允许快速结束。
SECOND_Q_FIRST_TEMPLATE_CONFIDENT_MAX_SHAPE = 0.30
SECOND_Q_FIRST_TEMPLATE_EARLY_FILL = 0.95         # 高填充、低重叠矩形可提前结束第一问搜索。
SECOND_Q_FIRST_TEMPLATE_EARLY_OVERLAP_RATIO = 0.02
SECOND_Q_FIRST_TEMPLATE_EARLY_MAX_BOUNDARY_MISSING = 0.15 # 外边界缺失过多时不得提前采用错误闭环。
SECOND_Q_FIRST_TEMPLATE_MIN_CLEARANCE_RATIO = 0.95 # 官方快路必须实际留出至少 95% 的目标机械间隔。
SECOND_Q_FIRST_TEMPLATE_PREFILTER_FILL = 0.90      # 闭环优化前只保留已经接近矩形的候选。
SECOND_Q_FIRST_TEMPLATE_PREFILTER_OVERLAP = 0.08  # 优化前允许少量裁剪/识别误差造成的重叠。

# 第二问边界/直角/接触评分参数；当前主评分与参考实现对齐，部分项只保留给调试或备用。
SECOND_Q_BOUNDARY_EDGE_WEIGHT = 80.0
SECOND_Q_BOUNDARY_EDGE_MAX_ERROR_PX = 28.0
SECOND_Q_BOUNDARY_COVER_DISTANCE_PX = 16.0
SECOND_Q_BOUNDARY_COVER_PARALLEL_MIN = 0.92
SECOND_Q_BOUNDARY_COVER_WEIGHT = 120.0
SECOND_Q_RIGHT_ANGLE_MIN_DEG = 75.0
SECOND_Q_RIGHT_ANGLE_MAX_DEG = 105.0
SECOND_Q_RIGHT_CORNER_WEIGHT = 220.0
SECOND_Q_RIGHT_CORNER_MISSING_PX = 80.0
SECOND_Q_RIGHT_CORNER_MAX_ERROR_PX = 28.0
SECOND_Q_RIGHT_CORNER_MIN_COUNT = 3
SECOND_Q_RECT_FROM_RIGHT_CORNERS = True
SECOND_Q_RECT_AREA_WEIGHT = 10.0

# 第二问固定 10 cm x 6 cm 目标矩形评分参数。
SECOND_Q_USE_FIXED_RECT_SCORE = True
SECOND_Q_FIXED_RECT_W_CM = FIRST_Q_RECT_W_CM
SECOND_Q_FIXED_RECT_H_CM = FIRST_Q_RECT_H_CM
SECOND_Q_FIXED_AREA_WEIGHT = 4.0       # 拼接面积接近固定目标面积的权重。
SECOND_Q_FIXED_RECT_AREA_WEIGHT = 30.0 # 最小外接矩形面积接近固定目标面积的权重。
SECOND_Q_FIXED_ASPECT_WEIGHT = 80000.0 # 长宽比偏离 10:6 的惩罚权重。
SECOND_Q_FIXED_PERIMETER_WEIGHT = 25.0 # 外轮廓周长偏离目标周长的惩罚权重。
SECOND_Q_BOUNDARY_REPAIR_REL_TOLERANCE = 0.12 # 边界缺口可由某条边修补时的长度容差。
SECOND_Q_BOUNDARY_REPAIR_WEIGHT = 160.0       # 边界修补项权重，当前主要保留备用。

# 第二问矩形硬约束：自由拼接候选必须先像一个矩形，再进入评分排序。
SECOND_Q_RECTLIKE_REJECT_ENABLED = True
SECOND_Q_RECTLIKE_MIN_FILL_RATIO = 0.88       # union 面积 / 最小外接矩形面积，低于它说明有坑或拼得散。
SECOND_Q_RECTLIKE_MAX_HULL_GAP_RATIO = 0.06   # 凸包比真实外轮廓多出的比例，尖顶/凹坑会明显变大。
SECOND_Q_RECTLIKE_MAX_CONTOUR_POINTS = 6      # 外轮廓近似点数上限；允许轻微噪声，但拒绝明显多边形。
SECOND_Q_RECTLIKE_MAX_ASPECT_ERROR = 0.22     # 长宽比相对 10:6 的 log 误差上限。

# 第二问调试输出和现场排查参数。
SECOND_Q_DEBUG_TOP_N = 5               # 打印/保留评分前几名组合。
SECOND_Q_DEBUG_CANDIDATES = False      # 是否打印全部候选边；现场运行关闭以避免刷屏拖慢。
SECOND_Q_DEBUG_MAX_CAND_PRINT = 60     # 每对碎片最多打印多少个候选。
SECOND_Q_SOLVE_PROGRESS_INTERVAL = 0   # 求解进度打印间隔；0 表示关闭，避免串口输出拖慢。
SECOND_Q_DEBUG_SHORT_EDGE_PX = 20.0    # 调试时提示短边的阈值，单位像素。
SECOND_Q_EDGE_CONTACT_MAX_ERROR_PX = 24.0
SECOND_Q_EDGE_CONTACT_WEIGHT = 20.0
SECOND_Q_HULL_GAP_WEIGHT = 80.0
SECOND_Q_TWO_PIECE_MAX_HULL_GAP_RATIO = 0.08
SECOND_Q_TWO_PIECE_MAX_OVERLAP_RATIO = 0.04
SECOND_Q_STRICT_BOUNDARY_REJECT = False
SECOND_Q_STRICT_EDGE_CONTACT_REJECT = False

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
AUTO_CAPTURE_ON_START = True           # 程序启动后是否自动执行一次稳定捕获并发送结果。
SEND_SERIAL_ON_KEY_PRESS = True        # 是否保留按键再次触发捕获/发送。
PERF_PROFILE = True                    # 是否统计每帧处理耗时。
TASK_MEMORY_GC_ENABLED = True          # 捕获/预览阶段结束后主动释放样本、求解缓存和循环引用。

# 稳定捕获参数；自动执行或按键触发后连续采集多帧，剔除离群值后输出稳定结果。
CAPTURE_FRAME_COUNT = 60               # 一次捕获最多处理帧数。
CAPTURE_HOLD_MS = 120000                # 捕获最长等待时间，单位毫秒。
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


# 视觉坐标到 STM32 坐标的补偿分为全局、抓取专用和放置专用三层。

# 全局比例以 A4 中心为基准，同时修正抓取和放置坐标的范围误差。
MECH_GLOBAL_X_SCALE = 1.06             # 左右都偏向中线时调大，偏向外侧时调小。
MECH_GLOBAL_Y_SCALE = 1.05             # 上下都偏向中线时调大，偏向外侧时调小。
MECH_GLOBAL_ROTATION_DEG = 1.62        # 整个机械坐标系绕 A4 中心旋转，+顺时针，-逆时针。

# 全局偏移同时移动抓取与放置坐标，用于手动对齐两套坐标系的整体原点偏差。
MECH_GLOBAL_FIXED_OFFSET_X_MM = -0.0    # 全部机械坐标沿 A4 透视图水平方向移动，+向右，单位 mm。
MECH_GLOBAL_FIXED_OFFSET_Y_MM = 4.5    # 全部机械坐标沿 A4 透视图垂直方向移动，+向下，单位 mm。


# 抓取/放置专用补偿只处理各自动作剩余的固定残差。
MECH_PICK_FIXED_OFFSET_X_MM = 0.0      # 抓取 X 偏心补偿归零，重新进行机械标定。
MECH_PICK_FIXED_OFFSET_Y_MM = 0.0      # 抓取 Y 偏心补偿归零，重新进行机械标定。
MECH_PLACE_FIXED_OFFSET_X_MM = 0.0     # 放置 X 固定补偿归零，重新进行机械标定。
MECH_PLACE_FIXED_OFFSET_Y_MM = 0.0     # 放置 Y 固定补偿归零，重新进行机械标定。
MECH_CALIBRATION_POINTS = [
    [0.0, 0.0],
    [A4_W_MM, 0.0],
    [A4_W_MM, A4_H_MM],
    [0.0, A4_H_MM],
]
# 推荐标定方式：直接填写 STM32 最终坐标系里的 A4 四角 TL/TR/BR/BL。
# 默认值与上面的旧标定加 XY 交换完全等价，因此启用后不会改变当前坐标。
MECH_DIRECT_OUTPUT_CALIBRATION_ENABLED = True
MECH_OUTPUT_CALIBRATION_POINTS = [
    [0.0, 0.0],
    [0.0, A4_W_MM],
    [A4_H_MM, A4_W_MM],
    [A4_H_MM, 0.0],
]

# 显示模式：0 原图 + A4 外框，1 透视图 + 最终碎片，2 色差图，3 raw mask，4 clean mask，5 候选轮廓，6 筛选结果 + 参数。
DISPLAY_MODE = 1
# 1 表示调试图直接显示原始摄像头坐标，避免透视插值把绿色窄缝显示成灰边。
DEBUG_SHOW_ORIGINAL_PROCESS = 1

# 调试视图保留给现场排查，常规阶段 2 显示由 DISPLAY_MODE 控制。
DEBUG_VIEW_MODE = 0

# 目标预览贴图：把左侧检测到的碎片图案也变换到右侧目标位置，便于检查花纹方向。
PREVIEW_TEXTURE_ENABLED = True
PREVIEW_TEXTURE_ALPHA = 0.30

# OpenCV 绘图颜色，顺序是 BGR，不是 RGB。
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

# 每块拼图使用不同的显示颜色（BGR）；目标矩形外框仍统一使用 CYAN。
PIECE_DISPLAY_COLORS = [
    (255, 0, 0),    # P0：蓝色
    (255, 0, 255),  # P1：品红
    (0, 165, 255),  # P2：橙色
    (0, 255, 255),  # P3：黄色
]


def piece_display_color(piece):
    """按稳定的碎片 ID 返回预览颜色，避免摆放顺序变化时颜色跳变。"""
    try:
        piece_id = int(piece.get("id", 0))
    except (TypeError, ValueError):
        piece_id = 0
    return PIECE_DISPLAY_COLORS[piece_id % len(PIECE_DISPLAY_COLORS)]

# 运行时状态和缓存，避免每帧重复计算固定模板。
capture_requested = AUTO_CAPTURE_ON_START  # 是否已经请求一次稳定捕获；启动自动执行时初始为 True。
first_question_area_ratios_cache = None     # 第一问模板面积比例缓存。
first_question_template_contour_cache = {}  # 第一问模板轮廓缓存。
first_question_target_layout_cache = {}     # 第一问目标布局缓存。
rotation_target_sample_cache = {}           # 旋转匹配采样点缓存。


def release_task_memory(capture_samples=None):
    """释放一次视觉任务不再需要的样本和求解缓存，不影响冻结结果预览。"""
    released_samples = 0
    if capture_samples is not None:
        released_samples = len(capture_samples)
        capture_samples.clear()
    released_rotation_cache = len(rotation_target_sample_cache)
    rotation_target_sample_cache.clear()
    if TASK_MEMORY_GC_ENABLED:
        gc.collect()
    return released_samples, released_rotation_cache


def close_runtime_resource(resource):
    """兼容不同 MaixPy 版本，安全关闭带 close/deinit 方法的运行资源。"""
    if resource is None:
        return
    for method_name in ("close", "deinit"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            pass
        break


# =========================
# Geometry
# =========================
# 基础几何工具：点排序、距离/角度、A4 透视矩阵、透视矫正。
# 这些函数不依赖题目逻辑，是后面检测、匹配、机械坐标换算的公共基础。

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
        if A4_GREEN_DETECTION_ENABLED:
            green_result, green_debug = self.detect_green_paper(frame)
            if green_result is not None:
                self.debug = green_debug
                return green_result

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
            "a4_method": "edge",
        }

    @staticmethod
    def detect_green_paper(frame):
        """优先由绿色纸的外轮廓定位 A4，碎片孔洞不会干扰最外层边界。"""
        frame_area = float(frame.shape[0] * frame.shape[1])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([PIECE_GREEN_H_LOW, PIECE_GREEN_S_LOW, 0], dtype=np.uint8)
        upper = np.array([PIECE_GREEN_H_HIGH, PIECE_GREEN_S_HIGH, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if A4_GREEN_MEDIAN_KERNEL >= 3:
            kernel = int(A4_GREEN_MEDIAN_KERNEL)
            if kernel % 2 == 0:
                kernel += 1
            mask = cv2.medianBlur(mask, kernel)
        close_kernel = np.ones(A4_GREEN_CLOSE_KERNEL, dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        candidate_count = 0
        for contour in contours:
            contour_area = abs(float(cv2.contourArea(contour)))
            area_ratio = contour_area / max(1.0, frame_area)
            if not A4_GREEN_MIN_AREA_RATIO <= area_ratio <= A4_GREEN_MAX_AREA_RATIO:
                continue
            hull = cv2.convexHull(contour)
            perimeter = float(cv2.arcLength(hull, True))
            if perimeter <= 1e-6:
                continue
            approx = None
            for epsilon_ratio in A4_GREEN_APPROX_EPSILONS:
                candidate = cv2.approxPolyDP(hull, perimeter * epsilon_ratio, True)
                if len(candidate) == 4 and cv2.isContourConvex(candidate):
                    approx = candidate
                    break
            if approx is None:
                continue

            corners = order_points(approx.reshape(4, 2))
            ratio = quad_aspect_ratio(corners)
            if not A4_RATIO_MIN <= ratio <= A4_RATIO_MAX:
                continue
            angles = quad_angles(corners)
            if not all(ANGLE_MIN <= angle <= ANGLE_MAX for angle in angles):
                continue

            quad_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, np.rint(corners).astype(np.int32), 255)
            quad_pixels = max(1, int(cv2.countNonZero(quad_mask)))
            green_pixels = int(cv2.countNonZero(cv2.bitwise_and(mask, quad_mask)))
            coverage_ratio = green_pixels / float(quad_pixels)
            if coverage_ratio < A4_GREEN_MIN_COVERAGE_RATIO:
                continue
            candidate_count += 1
            quad_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
            score = quad_area * (0.5 + coverage_ratio)
            if best is None or score > best[0]:
                best = (score, corners, area_ratio, coverage_ratio, ratio, angles)

        debug = {
            "method": "green",
            "contours": len(contours),
            "candidates": candidate_count,
        }
        if best is None:
            return None, debug
        _score, corners, area_ratio, coverage_ratio, ratio, angles = best
        debug.update({
            "best_area_ratio": area_ratio,
            "best_green_coverage": coverage_ratio,
            "best_aspect_ratio": ratio,
            "best_angles": angles,
        })
        return {
            "status": True,
            "corners": np.rint(corners).astype(np.int32).tolist(),
            "warp_size": [WARP_W, WARP_H],
            "midline": standard_midline(),
            "a4_method": "green",
            "a4_green_coverage": coverage_ratio,
        }, debug

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
        self.pending_edge_result = None
        self.pending_edge_frames = 0

    @staticmethod
    def corner_jump(result_a, result_b):
        corners_a = np.asarray(result_a.get("corners", []), dtype=np.float32).reshape(-1, 2)
        corners_b = np.asarray(result_b.get("corners", []), dtype=np.float32).reshape(-1, 2)
        if corners_a.shape != (4, 2) or corners_b.shape != (4, 2):
            return float("inf")
        return float(np.max(np.linalg.norm(corners_a - corners_b, axis=1)))

    def accept(self, result, smooth):
        accepted = result.copy()
        if smooth and self.last_result.get("status"):
            previous = np.asarray(self.last_result["corners"], dtype=np.float32)
            current = np.asarray(result["corners"], dtype=np.float32)
            blended = previous * (1.0 - A4_STABILIZER_ALPHA) + current * A4_STABILIZER_ALPHA
            accepted["corners"] = np.rint(blended).astype(np.int32).tolist()
            accepted["a4_smoothed"] = True
        accepted["stable"] = True
        accepted["lost_frames"] = 0
        self.last_result = accepted
        self.lost_frames = 0
        return self.last_result

    def hold_last(self, reason):
        self.lost_frames += 1
        if not self.last_result.get("status") or self.lost_frames > self.max_lost_frames:
            return {"status": False, "a4_reject_reason": reason}
        stable_result = self.last_result.copy()
        stable_result["stable"] = True
        stable_result["held"] = True
        stable_result["lost_frames"] = self.lost_frames
        stable_result["a4_reject_reason"] = reason
        return stable_result

    def update(self, result):
        if result.get("status"):
            if not self.last_result.get("status"):
                self.pending_edge_result = None
                self.pending_edge_frames = 0
                return self.accept(result, smooth=False)

            current_method = result.get("a4_method", "edge")
            last_method = self.last_result.get("a4_method", "edge")
            jump = self.corner_jump(self.last_result, result)

            # 绿色外轮廓是当前比赛纸的直接证据；恢复后立即接管，大跳变不参与平均。
            if current_method == "green":
                self.pending_edge_result = None
                self.pending_edge_frames = 0
                return self.accept(
                    result, smooth=jump <= A4_STABILIZER_MAX_CORNER_JUMP_PX)

            # 已稳定在绿色纸时，单帧 Canny 四边形不能直接覆盖它。
            if last_method == "green":
                if jump > A4_STABILIZER_MAX_CORNER_JUMP_PX:
                    self.pending_edge_result = None
                    self.pending_edge_frames = 0
                    return self.hold_last("edge_jump_from_green")
                if (
                    self.pending_edge_result is not None
                    and self.corner_jump(self.pending_edge_result, result)
                    <= A4_EDGE_SWITCH_TOLERANCE_PX
                ):
                    self.pending_edge_frames += 1
                else:
                    self.pending_edge_result = result.copy()
                    self.pending_edge_frames = 1
                self.pending_edge_result = result.copy()
                if self.pending_edge_frames < A4_EDGE_SWITCH_CONFIRM_FRAMES:
                    return self.hold_last("edge_wait_confirm")
                self.pending_edge_result = None
                self.pending_edge_frames = 0
                return self.accept(result, smooth=True)

            self.pending_edge_result = None
            self.pending_edge_frames = 0
            if jump <= A4_STABILIZER_MAX_CORNER_JUMP_PX:
                return self.accept(result, smooth=True)
            return self.hold_last("edge_corner_jump")

        self.pending_edge_result = None
        self.pending_edge_frames = 0
        if self.last_result.get("status") and self.lost_frames < self.max_lost_frames:
            return self.hold_last("detect_lost")

        self.last_result = {"status": False}
        self.lost_frames = 0
        return {"status": False}


# =========================
# Puzzle piece detector
# =========================

# =========================
# Piece Detection
# =========================
# 碎片检测主入口：在 A4 透视图中生成 mask、找轮廓、拟合多边形，
# 最后输出每块碎片的像素坐标、中心点、角度和调试信息。
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
        if not PIECE_ALLOW_A4_EDGE_TOUCH:
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

        use_convex_hull = PIECE_USE_CONVEX_HULL and not SECOND_QUESTION_MODE
        boundary = cv2.convexHull(contour) if use_convex_hull else contour
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
    elif SECOND_QUESTION_MODE:
        for piece_id, piece in enumerate(pieces):
            piece["template"] = "P%d" % piece_id

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

    # 模式 1 也缓存透视后的原始 mask 轮廓，用于红蓝轮廓同屏对照。
    if DISPLAY_MODE == 6 or PIECE_DEBUG_OVERLAY_RAW_CONTOUR:
        debug["accepted_source_contours"] = warp_contours(accepted_source_contours, matrix)
        if DISPLAY_MODE == 6 and DEBUG_SHOW_ORIGINAL_PROCESS:
            debug["accepted_source_contours_original"] = accepted_source_contours

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


# 根据当前配置选择碎片分割方式。black/hsv/lab 三套方法共用同一套后处理流程。
def make_piece_mask(frame, detect_mask):
    if PIECE_MASK_METHOD == "hsv":
        return make_piece_mask_hsv(frame, detect_mask)
    if PIECE_MASK_METHOD == "black":
        return make_piece_mask_black(frame, detect_mask)
    return make_piece_mask_lab(frame, detect_mask)


def make_piece_mask_hsv(frame, detect_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # 只有色相和饱和度都明确符合绿色纸特征时才判为背景。
    # 白色、灰色、阴影和扑克深色花纹会因饱和度较低而保留。
    green_background = (
        (h >= PIECE_GREEN_H_LOW)
        & (h <= PIECE_GREEN_H_HIGH)
        & (s >= PIECE_GREEN_S_LOW)
        & (s <= PIECE_GREEN_S_HIGH)
    )
    piece_mask = np.where(green_background, 0, 255).astype(np.uint8)
    piece_mask = cv2.bitwise_and(piece_mask, detect_mask)
    if PIECE_HSV_MEDIAN_ENABLED:
        piece_mask = smooth_piece_mask(piece_mask, detect_mask)

    # 模式 2 显示滤波前的绿色分类，白色表示非绿色。
    distance_map = np.where(green_background, 0.0, 255.0).astype(np.float32)
    distance_map *= detect_mask.astype(np.float32) / 255.0
    bg_color = np.array([PIECE_GREEN_H_LOW, PIECE_GREEN_S_LOW, PIECE_GREEN_V_LOW], dtype=np.float32)
    return distance_map, piece_mask, bg_color


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


# =========================
# First Question Solver
# =========================
# 第一问：把检测到的碎片和固定 A/B/C/D 模板匹配，
# 再计算每块应该移动到 10 cm x 6 cm 目标矩形中的哪个位置。
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


def second_question_first_template_match(pieces):
    """第二问中识别第一问官方 A/B/C/D 切法,命中时返回最佳模板分配。"""
    if (
        not SECOND_Q_FIRST_TEMPLATE_DETECT_ENABLED
        or len(pieces) != len(FIRST_Q_TEMPLATES)
    ):
        return None

    template_ratios = cached_first_question_template_area_ratios()
    piece_infos = []
    total_area = 0.0
    for piece in pieces:
        polygon = np.asarray(piece.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3:
            return None
        contour = polygon.reshape(-1, 1, 2)
        area = float(piece.get("area", 0))
        if area <= 0.0:
            area = abs(float(cv2.contourArea(contour)))
        total_area += max(0.0, area)
        scores = first_question_template_scores(contour)
        piece_infos.append({
            "area": max(0.0, area),
            "points": int(piece.get("points", len(polygon))),
            "scores": scores,
        })
    total_area = max(1.0, total_area)

    best = None
    for template_indices in itertools.permutations(range(len(FIRST_Q_TEMPLATES))):
        cost = 0.0
        max_shape_score = 0.0
        details = []
        for piece_index, template_index in enumerate(template_indices):
            info = piece_infos[piece_index]
            shape_score = 999.0
            for score in info["scores"]:
                if score["index"] == template_index:
                    shape_score = float(score["shape_score"])
                    break
            area_ratio = info["area"] / total_area
            area_score = template_area_match_score(area_ratio, template_ratios[template_index])
            point_score = abs(info["points"] - len(FIRST_Q_TEMPLATES[template_index]["polygon_cm"]))
            item_cost = (
                shape_score
                + SECOND_Q_FIRST_TEMPLATE_AREA_WEIGHT * min(area_score, 0.60)
                + SECOND_Q_FIRST_TEMPLATE_POINT_WEIGHT * point_score
            )
            cost += item_cost
            max_shape_score = max(max_shape_score, shape_score)
            details.append({
                "piece": piece_index,
                "template_index": template_index,
                "template": FIRST_Q_TEMPLATES[template_index]["name"],
                "shape_score": shape_score,
                "area_ratio": area_ratio,
                "area_score": area_score,
                "point_score": point_score,
                "cost": item_cost,
            })
        if best is None or cost < best["cost"]:
            best = {
                "assignment": template_indices,
                "cost": cost,
                "max_shape_score": max_shape_score,
                "details": details,
            }

    if best is None:
        return None
    if best["cost"] > SECOND_Q_FIRST_TEMPLATE_MAX_COST:
        return None
    if best["max_shape_score"] > SECOND_Q_FIRST_TEMPLATE_MAX_SHAPE_SCORE:
        return None
    return best



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


def rigid_transform(angle_rad, tx, ty):
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.array(
        [
            [cosine, -sine, float(tx)],
            [sine, cosine, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def apply_homography_points(points, transform):
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(points_array), 1), dtype=np.float32)
    homogeneous = np.hstack((points_array, ones)).dot(np.asarray(transform, dtype=np.float32).T)
    return homogeneous[:, :2] / np.maximum(1e-6, homogeneous[:, 2:3])


def align_edge_transform(src_a, src_b, dst_a, dst_b):
    source_vec = np.asarray(src_b, dtype=np.float32) - np.asarray(src_a, dtype=np.float32)
    target_vec = np.asarray(dst_b, dtype=np.float32) - np.asarray(dst_a, dtype=np.float32)
    angle = math.atan2(float(target_vec[1]), float(target_vec[0])) - math.atan2(float(source_vec[1]), float(source_vec[0]))
    transform = rigid_transform(angle, 0.0, 0.0)
    mapped = apply_homography_points(np.asarray([src_a], dtype=np.float32), transform)[0]
    transform[:2, 2] = np.asarray(dst_a, dtype=np.float32) - mapped
    return transform


# =========================
# Second Question Solver
# =========================
# 第二问：不依赖固定模板，使用碎片边长、T 形局部边、复合边、
# 光栅化矩形评分来枚举并筛选最可能的拼接方案。
def second_question_target_origin(target_side=None, target_size=None):
    target_side = target_side or SECOND_Q_TARGET_SIDE
    if target_side == "auto":
        target_side = "left"
    if target_size is None:
        rect_w = (SECOND_Q_RECT_MIN_W_CM + SECOND_Q_RECT_MAX_W_CM) * 0.5 * (WARP_W - 1) / A4_W_CM
        rect_h = (SECOND_Q_RECT_MIN_H_CM + SECOND_Q_RECT_MAX_H_CM) * 0.5 * (WARP_H - 1) / A4_H_CM
    else:
        rect_w = float(target_size[0])
        rect_h = float(target_size[1])
    gap_px = SECOND_Q_CENTERLINE_GAP_CM * (WARP_W - 1) / A4_W_CM
    center_x = WARP_W * 0.5
    y = (WARP_H - rect_h) * 0.5
    if target_side == "right":
        x = center_x + gap_px
    else:
        x = center_x - gap_px - rect_w
    x = max(0.0, min(float(x), WARP_W - rect_w - 1.0))
    y = max(0.0, min(float(y), WARP_H - rect_h - 1.0))
    return np.asarray([x, y], dtype=np.float32)


def second_question_target_available_size(target_side=None):
    target_side = target_side or SECOND_Q_TARGET_SIDE
    if target_side == "auto":
        target_side = "left"
    gap_px = SECOND_Q_CENTERLINE_GAP_CM * (WARP_W - 1) / A4_W_CM
    center_x = WARP_W * 0.5
    if target_side == "right":
        available_w = WARP_W - (center_x + gap_px) - 2.0
    else:
        available_w = center_x - gap_px - 2.0
    return np.asarray(
        [max(1.0, float(available_w)), max(1.0, float(WARP_H - 2.0))],
        dtype=np.float32,
    )


def choose_second_question_target_side(pieces):
    if SECOND_Q_TARGET_SIDE in ("left", "right"):
        return SECOND_Q_TARGET_SIDE
    centers = [piece.get("center") for piece in pieces if piece.get("center")]
    if not centers:
        return "left"
    mean_x = float(np.mean([center[0] for center in centers]))
    return "left" if mean_x >= WARP_W * 0.5 else "right"


def second_question_target_rect(target_side=None, target_size=None):
    origin = second_question_target_origin(target_side, target_size)
    if target_size is None:
        rect_w = (SECOND_Q_RECT_MIN_W_CM + SECOND_Q_RECT_MAX_W_CM) * 0.5 * (WARP_W - 1) / A4_W_CM
        rect_h = (SECOND_Q_RECT_MIN_H_CM + SECOND_Q_RECT_MAX_H_CM) * 0.5 * (WARP_H - 1) / A4_H_CM
    else:
        rect_w = float(target_size[0])
        rect_h = float(target_size[1])
    return np.asarray(
        [
            origin,
            origin + [rect_w, 0.0],
            origin + [rect_w, rect_h],
            origin + [0.0, rect_h],
        ],
        dtype=np.float32,
    )


def second_question_edge_components(edge_ref, point_count=None):
    if isinstance(edge_ref, tuple):
        start, span = edge_ref
        return [int((start + offset) % point_count) for offset in range(int(span))]
    return [int(edge_ref)]


def second_question_edge_usage_intervals(edge_ref, point_count, start_t, end_t):
    start_t = max(0.0, min(1.0, float(start_t)))
    end_t = max(0.0, min(1.0, float(end_t)))
    if end_t < start_t:
        start_t, end_t = end_t, start_t
    if isinstance(edge_ref, tuple):
        return [
            (int(component), 0.0, 1.0)
            for component in second_question_edge_components(edge_ref, point_count)
        ]
    return [(int(edge_ref), start_t, end_t)]


def second_question_interval_overlaps(a0, a1, b0, b1):
    return max(float(a0), float(b0)) < min(float(a1), float(b1)) - 1e-4


def second_question_can_claim_edge_intervals(used, piece_index, intervals):
    for component, start_t, end_t in intervals:
        for used_start, used_end in used.get((piece_index, component), []):
            if second_question_interval_overlaps(start_t, end_t, used_start, used_end):
                return False
    return True


def second_question_claim_edge_intervals(used, piece_index, intervals):
    for component, start_t, end_t in intervals:
        used.setdefault((piece_index, component), []).append((start_t, end_t))


def second_question_edge_label(edge_ref):
    if isinstance(edge_ref, tuple):
        return "E%d+%d" % (int(edge_ref[0]), int(edge_ref[1]))
    return "E%d" % int(edge_ref)


def second_question_resolve_edge(points, edge_ref):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if isinstance(edge_ref, tuple):
        start, span = edge_ref
        return points[int(start) % len(points)], points[(int(start) + int(span)) % len(points)]
    return polygon_edges(points)[int(edge_ref)]


def second_question_point_line_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    line_len = float(np.linalg.norm(line_vec))
    if line_len <= 1e-6:
        return float(np.linalg.norm(point - line_start))
    point_vec = point - line_start
    cross = abs(float(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]))
    return cross / line_len


def second_question_piece_edge_refs(piece):
    points = np.asarray(piece, dtype=np.float32).reshape(-1, 2)
    count = len(points)
    refs = []
    for edge_index, edge in enumerate(polygon_edges(points)):
        refs.append((edge_index, edge))

    max_span = min(int(SECOND_Q_COMPOSITE_EDGE_MAX_SPAN), max(1, count - 1))
    for span in range(2, max_span + 1):
        for start in range(count):
            end = (start + span) % count
            chord_start = points[start]
            chord_end = points[end]
            chord_len = float(np.linalg.norm(chord_end - chord_start))
            path_len = 0.0
            max_deviation = 0.0
            for offset in range(span):
                edge_start = points[(start + offset) % count]
                edge_end = points[(start + offset + 1) % count]
                path_len += float(np.linalg.norm(edge_end - edge_start))
                if 0 < offset < span:
                    mid_point = points[(start + offset) % count]
                    max_deviation = max(
                        max_deviation,
                        second_question_point_line_distance(mid_point, chord_start, chord_end),
                    )
            if chord_len <= 1e-6 or path_len <= 1e-6:
                continue
            if chord_len / path_len < SECOND_Q_COMPOSITE_EDGE_MIN_CHORD_RATIO:
                continue
            if max_deviation > SECOND_Q_COMPOSITE_EDGE_MAX_DEVIATION_PX:
                continue
            refs.append(((start, span), (chord_start, chord_end)))
    return refs


def second_question_polygon_angles(polygon):
    """每个顶点的内角(弧度),对凹顶点(reflex)同样正确。"""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    count = len(points)
    # 手写有符号面积:MaixPy 的 cv2.contourArea 不一定支持 oriented 参数。
    x = points[:, 0]
    y = points[:, 1]
    signed = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    orientation = 1.0 if signed >= 0 else -1.0
    angles = []
    for index in range(count):
        edge_prev = points[index] - points[index - 1]
        edge_next = points[(index + 1) % count] - points[index]
        turn = math.atan2(
            float(edge_prev[0] * edge_next[1] - edge_prev[1] * edge_next[0]),
            float(edge_prev[0] * edge_next[0] + edge_prev[1] * edge_next[1]),
        )
        angle = math.pi - orientation * turn
        if angle <= 0:
            angle += 2 * math.pi
        elif angle > 2 * math.pi:
            angle -= 2 * math.pi
        angles.append(angle)
    return np.asarray(angles, dtype=float)


def second_question_endpoint_angle_bad(angle_sum, tolerance=math.radians(12.0)):
    """端点处两片凸片内角和落在 (102°,168°):物理上不能是边界(90/180°)
    也不能是三岔节点(>180°),只可能是四岔节点 —— 作为可疑信号累计。"""
    return math.pi / 2 + tolerance < angle_sum < math.pi - tolerance


def second_question_edge_endpoint_vertices(edge_ref, point_count):
    """返回边的(起点顶点索引, 终点顶点索引),复合边取弦的两端。"""
    if isinstance(edge_ref, tuple):
        start, span = edge_ref
        return int(start) % point_count, (int(start) + int(span)) % point_count
    return int(edge_ref) % point_count, (int(edge_ref) + 1) % point_count


# 生成候选边匹配：包含完整边匹配、T 形局部边匹配，以及多段复合边匹配。
# 候选只表示“可能相邻”，最终真假由全局矩形评分决定。
def second_question_candidate_matchings(pieces):
    all_edges = {}
    for piece_index, piece in enumerate(pieces):
        for edge_ref, edge in second_question_piece_edge_refs(piece):
            all_edges[(piece_index, edge_ref)] = edge

    vertex_angles = [second_question_polygon_angles(piece) for piece in pieces]
    convex = [
        bool(np.all(angles <= math.pi + math.radians(8.0)))
        for angles in vertex_angles
    ]
    edge_lengths = {
        key: float(np.linalg.norm(value[1] - value[0]))
        for key, value in all_edges.items()
    }

    candidates = []
    for (piece_i, edge_i), (piece_j, edge_j) in itertools.combinations(all_edges, 2):
        if piece_i == piece_j:
            continue
        len_a = edge_lengths[(piece_i, edge_i)]
        len_b = edge_lengths[(piece_j, edge_j)]
        if len_a <= 1e-6 or len_b <= 1e-6:
            continue
        rel = abs(len_a - len_b) / max(len_a, len_b)
        both_convex = convex[piece_i] and convex[piece_j]
        count_i = len(pieces[piece_i])
        count_j = len(pieces[piece_j])
        start_i, end_i = second_question_edge_endpoint_vertices(edge_i, count_i)
        start_j, end_j = second_question_edge_endpoint_vertices(edge_j, count_j)

        def bad_ends(*pairs):
            # 角度互补仅作排序软惩罚:(102°,168°) 的和也可能是合法的三片
            # 边界节点/四岔内部节点,绝不能硬过滤,由评分函数最终裁决。
            if not both_convex:
                return 0
            return sum(
                second_question_endpoint_angle_bad(
                    vertex_angles[ii][kk % len(vertex_angles[ii])]
                    + vertex_angles[jj][ll % len(vertex_angles[jj])]
                )
                for ii, kk, jj, ll in pairs
            )

        # 反向对接:j 边起点 c ↔ i 边终点 b,j 边终点 d ↔ i 边起点 a
        if rel < SECOND_Q_MATCH_REL_TOLERANCE:
            bad = bad_ends(
                (piece_i, end_i, piece_j, start_j),
                (piece_i, start_i, piece_j, end_j),
            )
            candidates.append((rel + 0.02 * bad, piece_i, edge_i, piece_j, edge_j, 0.0, 1.0, 0.0, 1.0))

        def partial_penalty(ii, kk, jj, ll):
            # partial 的顶点-顶点接触端:真实 T 节点该端在两片外边界上,
            # 内角和 ≈180°(或 90° 卡角);偏离越多越可疑(连续,上限 0.1)。
            if not both_convex:
                return 0.15
            angle_sum = (
                vertex_angles[ii][kk % len(vertex_angles[ii])]
                + vertex_angles[jj][ll % len(vertex_angles[jj])]
            )
            if angle_sum >= math.pi - math.radians(12.0):
                dev = abs(math.pi - angle_sum)  # 略超 180° 同样收小惩罚
            else:
                dev = min(abs(angle_sum - math.pi), abs(angle_sum - math.pi / 2))
                dev = max(0.0, dev - math.radians(12.0))
            return 0.15 + min(0.10, 0.005 * math.degrees(dev))

        ratio = min(len_a, len_b) / max(len_a, len_b)
        if SECOND_Q_PARTIAL_RATIO_MIN <= ratio <= SECOND_Q_PARTIAL_RATIO_MAX:
            # T 节点:短边整条占据长边的 [0,ratio] 或 [1-ratio,1]。
            if len_a > len_b:
                # j 整条边占据 i 边的 [0,ratio](边界端 a↔d)或 [1-ratio,1](b↔c)
                candidates.append((
                    partial_penalty(piece_i, start_i, piece_j, end_j)
                    + 0.02 * bad_ends((piece_i, start_i, piece_j, end_j)),
                    piece_i, edge_i, piece_j, edge_j, 0.0, ratio, 0.0, 1.0,
                ))
                candidates.append((
                    partial_penalty(piece_i, end_i, piece_j, start_j)
                    + 0.02 * bad_ends((piece_i, end_i, piece_j, start_j)),
                    piece_i, edge_i, piece_j, edge_j, 1.0 - ratio, 1.0, 0.0, 1.0,
                ))
            else:
                candidates.append((
                    partial_penalty(piece_i, end_i, piece_j, start_j)
                    + 0.02 * bad_ends((piece_i, end_i, piece_j, start_j)),
                    piece_i, edge_i, piece_j, edge_j, 0.0, 1.0, 0.0, ratio,
                ))
                candidates.append((
                    partial_penalty(piece_i, start_i, piece_j, end_j)
                    + 0.02 * bad_ends((piece_i, start_i, piece_j, end_j)),
                    piece_i, edge_i, piece_j, edge_j, 0.0, 1.0, 1.0 - ratio, 1.0,
                ))

    # 多段部分边匹配(双T/多T切割):一条长边可能被两条短边分段共享
    for (piece_i, edge_i), _edge in all_edges.items():
        len_a = edge_lengths[(piece_i, edge_i)]
        if len_a <= 1e-6:
            continue
        short_edges = []
        for (piece_j, edge_j), _other in all_edges.items():
            if piece_i == piece_j:
                continue
            len_b = edge_lengths[(piece_j, edge_j)]
            ratio = len_b / len_a
            # 短边长度应该在长边的 10% ~ 90% 之间
            if 0.10 <= ratio <= 0.90:
                short_edges.append((piece_j, edge_j, len_b))
        if len(short_edges) < 2:
            continue
        for (piece_j1, edge_j1, len_b1), (piece_j2, edge_j2, len_b2) in itertools.combinations(short_edges, 2):
            total_short = len_b1 + len_b2
            length_rel = abs(total_short - len_a) / len_a
            if length_rel > 0.15:  # 15% 容差
                continue
            # 分割点:短边1占 [0, len_b1/len_a],短边2占 [len_b1/len_a, 1]
            split = len_b1 / len_a
            candidates.append((
                0.15 + length_rel,
                piece_i, edge_i, piece_j1, edge_j1, 0.0, split, 0.0, 1.0,
            ))
            candidates.append((
                0.15 + length_rel,
                piece_i, edge_i, piece_j2, edge_j2, split, 1.0, 0.0, 1.0,
            ))

    # edge_ref 可能是复合边元组,不能直接按元组比较,一律按惩罚排序。
    candidates.sort(key=lambda item: item[0])
    # Preserve ambiguity *per pair of pieces*. A global top-N shortlist is
    # biased toward repeated outer-card lengths (especially rectangular or
    # near-symmetric fragments) and can discard the only true cut edge for
    # another pair. That leaves the global scorer no correct topology to
    # choose from, regardless of how strongly overlap is penalized.
    grouped = {}
    for candidate in candidates:
        pair = tuple(sorted((candidate[1], candidate[3])))
        grouped.setdefault(pair, []).append(candidate)

    shortlist = []
    for group in grouped.values():
        full = [candidate for candidate in group if tuple(candidate[5:]) == (0.0, 1.0, 0.0, 1.0)]
        partial = [candidate for candidate in group if tuple(candidate[5:]) != (0.0, 1.0, 0.0, 1.0)]
        shortlist.extend(full[:SECOND_Q_FULL_MATCHES_PER_PAIR])
        shortlist.extend(partial[:SECOND_Q_PARTIAL_MATCHES_PER_PAIR])
    shortlist.sort(key=lambda item: item[0])
    return shortlist


def second_question_match_segments(pieces, match):
    _, piece_i, edge_i, piece_j, edge_j, ia0, ia1, ja0, ja1 = match
    a, b = second_question_resolve_edge(pieces[piece_i], edge_i)
    c, d = second_question_resolve_edge(pieces[piece_j], edge_j)
    return (
        a + (b - a) * ia0,
        a + (b - a) * ia1,
        c + (d - c) * ja0,
        c + (d - c) * ja1,
    )


# 从候选边中枚举可连通的组合。这里用惩罚预算和 heap 只保留前 max_scored 个组合，
# 避免在 MaixCAM 上一次性保存过多组合导致内存压力。
def second_question_matching_sets(
    pieces,
    candidates=None,
    cut_mode="auto",
    max_scored=SECOND_Q_MAX_SCORED_COMBOS,
    allowed_pairs=None,
    pair_candidate_limit=None,
    tree_only=False,
    cycle_only=False,
    candidate_priorities=None,
):
    """按碎片对拓扑枚举匹配组合:full/partial 任意混合,|E| = N-1 或 N。

    邻接对偶图可以是树(N-1)或带一个环(N,如共顶点族/双T链),环上的
    closure_error 由评分函数裁决,不再在组合阶段按 cut_mode 硬编码结构。
    按"碎片对拓扑"枚举 + 边区间冲突提前剪枝 + 惩罚预算剪枝,避免在全
    候选集上做组合爆炸。
    """
    count = len(pieces)
    if count == 1:
        yield ()
        return
    if candidates is None:
        candidates = second_question_candidate_matchings(pieces)
    allowed_pairs = None if allowed_pairs is None else {tuple(sorted(pair)) for pair in allowed_pairs}
    by_pair = {}
    for match in candidates:
        pair = tuple(sorted((match[1], match[3])))
        if allowed_pairs is not None and pair not in allowed_pairs:
            continue
        group = by_pair.setdefault(pair, [])
        group.append(match)
    for group in by_pair.values():
        group.sort(key=lambda item: (
            candidate_priorities.get(item, float(item[0]))
            if candidate_priorities is not None else float(item[0])
        ))
        if pair_candidate_limit is not None:
            del group[int(pair_candidate_limit):]
    # 预计算每个候选的边区间占用,避免递归内重复调用辅助函数(枚举是耗时大头)
    active_candidates = [match for matches in by_pair.values() for match in matches]
    claims_cache = {}
    for match in active_candidates:
        _, piece_i, edge_i, piece_j, edge_j, ia0, ia1, ja0, ja1 = match
        claims = [
            (piece_i, component, start_t, end_t)
            for component, start_t, end_t in second_question_edge_usage_intervals(
                edge_i, len(pieces[piece_i]), ia0, ia1)
        ]
        claims += [
            (piece_j, component, start_t, end_t)
            for component, start_t, end_t in second_question_edge_usage_intervals(
                edge_j, len(pieces[piece_j]), ja0, ja1)
        ]
        claims_cache[match] = claims
    pair_list = sorted(by_pair)

    def connected(topo):
        degree = [0] * count
        graph = [set() for _ in range(count)]
        for piece_i, piece_j in topo:
            degree[piece_i] += 1
            degree[piece_j] += 1
            graph[piece_i].add(piece_j)
            graph[piece_j].add(piece_i)
        if any(item == 0 for item in degree):
            return False
        seen, stack = {0}, [0]
        while stack:
            for neighbor in graph[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == count

    if cycle_only:
        topology_sizes = (count,)
    elif tree_only:
        topology_sizes = (count - 1,)
    else:
        topology_sizes = (count - 1, count)
    topologies = []
    for size in topology_sizes:
        if size < 1 or size > len(pair_list):
            continue
        topologies.extend(
            topo for topo in itertools.combinations(pair_list, size)
            if connected(topo)
        )

    def match_penalty(match):
        if candidate_priorities is not None:
            return float(candidate_priorities.get(match, float(match[0])))
        return float(match[0])

    def state_penalty(topo, indices):
        return sum(match_penalty(by_pair[pair][index]) for pair, index in zip(topo, indices))

    def combo_has_conflict(combo):
        used = {}
        for match in combo:
            for piece_index, component, start_t, end_t in claims_cache[match]:
                for used_start, used_end in used.get((piece_index, component), ()):
                    if max(start_t, used_start) < min(end_t, used_end) - 1e-4:
                        return True
                used.setdefault((piece_index, component), []).append((start_t, end_t))
        return False

    # 每个拓扑是一组有序候选列表的笛卡尔积。用 best-first 同时推进全部
    # 拓扑，直接按总惩罚升序产生合法组合，避免先穷举再截取前 max_scored。
    frontier = []
    visited = set()
    sequence = 0
    for topology_index, topo in enumerate(topologies):
        if any(not by_pair[pair] for pair in topo):
            continue
        indices = (0,) * len(topo)
        penalty = state_penalty(topo, indices)
        if penalty > SECOND_Q_COMBO_PENALTY_BUDGET:
            continue
        heapq.heappush(frontier, (penalty, sequence, topology_index, indices))
        visited.add((topology_index, indices))
        sequence += 1

    yielded = 0
    checked = 0
    while frontier and yielded < max_scored:
        _penalty, _sequence, topology_index, indices = heapq.heappop(frontier)
        checked += 1
        if SECOND_Q_MAX_MATCHING_COMBO_CHECKS > 0 and checked > SECOND_Q_MAX_MATCHING_COMBO_CHECKS:
            break
        topo = topologies[topology_index]
        combo = tuple(by_pair[pair][index] for pair, index in zip(topo, indices))
        if not combo_has_conflict(combo):
            yielded += 1
            yield combo

        for dimension, pair in enumerate(topo):
            next_index = indices[dimension] + 1
            if next_index >= len(by_pair[pair]):
                continue
            next_indices = list(indices)
            next_indices[dimension] = next_index
            next_indices = tuple(next_indices)
            state_key = (topology_index, next_indices)
            if state_key in visited:
                continue
            next_penalty = state_penalty(topo, next_indices)
            if next_penalty > SECOND_Q_COMBO_PENALTY_BUDGET:
                continue
            visited.add(state_key)
            heapq.heappush(
                frontier,
                (next_penalty, sequence, topology_index, next_indices),
            )
            sequence += 1



def second_question_reject(reject_stats, reason):
    if reject_stats is not None:
        reject_stats[reason] = reject_stats.get(reason, 0) + 1
    return None


def second_question_format_reject_stats(reject_stats):
    if not reject_stats:
        return "none"
    return ",".join("%s=%d" % (key, reject_stats[key]) for key in sorted(reject_stats))


def second_question_candidate_clearance_ok(result, matches, reject_stats, strict_gap=False):
    """最终候选必须能通过平移消除重叠；官方模板还要求基本达到计划间距。"""
    if result is None or SECOND_Q_TARGET_GAP_MM <= 0.0:
        return result is not None
    px_per_mm = 0.5 * (
        (WARP_W - 1.0) / A4_W_MM
        + (WARP_H - 1.0) / A4_H_MM
    )
    requested_gap_px = float(SECOND_Q_TARGET_GAP_MM) * px_per_mm
    _offsets, _shifted, achieved_gap_px, clearance_overlap = (
        second_question_apply_clearance(result[2], matches, requested_gap_px)
    )
    clearance_ratio = achieved_gap_px / max(1e-6, requested_gap_px)
    result[3]["clearance_ratio"] = clearance_ratio
    result[3]["clearance_overlap"] = clearance_overlap
    if clearance_overlap > 1.0:
        second_question_reject(reject_stats, "clearance_overlap")
        return False
    if strict_gap and clearance_ratio < SECOND_Q_FIRST_TEMPLATE_MIN_CLEARANCE_RATIO:
        second_question_reject(reject_stats, "template_clearance")
        return False
    return True


def second_question_rectlike_reject_reason(contour_world, rect_area, union_area, hull_area, aspect, fixed_aspect):
    if not SECOND_Q_RECTLIKE_REJECT_ENABLED:
        return None
    if rect_area <= 1e-6 or union_area <= 1e-6:
        return "矩形_空"

    fill_ratio = float(union_area) / max(1.0, float(rect_area))
    if fill_ratio < SECOND_Q_RECTLIKE_MIN_FILL_RATIO:
        return "矩形_填充率低"

    hull_gap_ratio = max(0.0, float(hull_area) - float(union_area)) / max(1.0, float(union_area))
    if hull_gap_ratio > SECOND_Q_RECTLIKE_MAX_HULL_GAP_RATIO:
        return "矩形_凸包间隙大"

    contour = np.asarray(contour_world, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return "矩形_周长为零"
    approx = cv2.approxPolyDP(contour, SECOND_Q_RECT_APPROX_EPSILON_RATIO * perimeter, True)
    if len(approx) > SECOND_Q_RECTLIKE_MAX_CONTOUR_POINTS:
        return "矩形_轮廓点过多"

    # 长宽比受碎片缺角、阴影缝和 minAreaRect 方向影响很大；这里不能一票否决。
    # 具体偏离量继续在 second_question_score_assembly() 里作为软评分项处理。
    return None


def second_question_pose_graph_closure_error(pieces, matches, transforms):
    """返回全部已声明接缝在当前位姿下的平均端点误差（像素）。"""
    errors = []
    for match in matches:
        _, piece_i, _edge_i, piece_j, _edge_j = match[:5]
        ia, ib, ja, jb = second_question_match_segments(pieces, match)
        world_i = apply_homography_points(np.asarray([ia, ib], dtype=np.float32), transforms[piece_i])
        world_j = apply_homography_points(np.asarray([jb, ja], dtype=np.float32), transforms[piece_j])
        errors.extend(np.linalg.norm(world_i - world_j, axis=1).tolist())
    return float(np.mean(errors)) if errors else 0.0

def second_question_assemble_from_matches(
    pieces,
    matches,
    reject_stats=None,
    optimize_cycle=False,
    use_fixed_rect_score=None,
    mask_scale=0.5,
    coarse_score=False,
    geometry_score=False,
):
    adjacency = [[] for _ in pieces]
    for match in matches:
        _, piece_i, _edge_i, piece_j, _edge_j = match[:5]
        adjacency[piece_i].append((piece_j, match, False))
        adjacency[piece_j].append((piece_i, match, True))

    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3, dtype=np.float32)
    stack = [0]
    closure_error = 0.0
    while stack:
        piece_i = stack.pop()
        for piece_j, match, reversed_sides in adjacency[piece_i]:
            ia, ib, ja, jb = second_question_match_segments(pieces, match)
            if reversed_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            world_a, world_b = apply_homography_points(np.asarray([ia, ib], dtype=np.float32), transforms[piece_i])
            proposed = align_edge_transform(ja, jb, world_b, world_a)
            if transforms[piece_j] is None:
                transforms[piece_j] = proposed
                stack.append(piece_j)
            else:
                previous = apply_homography_points(pieces[piece_j], transforms[piece_j])
                current = apply_homography_points(pieces[piece_j], proposed)
                closure_error += float(np.linalg.norm(current - previous, axis=1).mean())

    if any(transform is None for transform in transforms):
        return second_question_reject(reject_stats, "disconnected")
    assembled = [apply_homography_points(piece, transform) for piece, transform in zip(pieces, transforms)]
    if optimize_cycle and len(matches) >= len(pieces):
        # 先用未优化位姿做便宜预筛，避免给明显错误的闭环运行 LM。
        raw_result = second_question_score_assembly(
            transforms, assembled, matches, closure_error, reject_stats, apply_rectlike_reject=False, use_fixed_rect_score=use_fixed_rect_score)
        if raw_result is None:
            return None
        raw_detail = raw_result[3]
        raw_union = max(1.0, float(raw_detail.get("union_area", 1.0)))
        raw_overlap_ratio = float(raw_detail.get("overlap", 0.0)) / raw_union
        if (
            float(raw_detail.get("fill", 0.0)) < SECOND_Q_FIRST_TEMPLATE_PREFILTER_FILL
            or raw_overlap_ratio > SECOND_Q_FIRST_TEMPLATE_PREFILTER_OVERLAP
        ):
            return second_question_reject(reject_stats, "template_cycle_prefilter")
        transforms = second_question_optimize_pose_graph(pieces, matches, transforms)
        assembled = [
            apply_homography_points(piece, transform)
            for piece, transform in zip(pieces, transforms)
        ]
        closure_error = second_question_pose_graph_closure_error(pieces, matches, transforms)
    if geometry_score:
        return second_question_score_assembly_fast(
            transforms, assembled, matches, closure_error, reject_stats,
            use_fixed_rect_score=use_fixed_rect_score,
            coarse_score=coarse_score,
        )
    return second_question_score_assembly(
        transforms, assembled, matches, closure_error, reject_stats,
        use_fixed_rect_score=use_fixed_rect_score,
        mask_scale=mask_scale,
        coarse_score=coarse_score,
    )


def second_question_boundary_edge_error(assembled, rect, matches):
    rect_points = order_points(cv2.boxPoints(rect))
    rect_sides = polygon_edges(rect_points)
    matched_edges = set()
    for match in matches:
        _, piece_i, edge_i, piece_j, edge_j = match[:5]
        for component in second_question_edge_components(edge_i, len(assembled[piece_i])):
            matched_edges.add((piece_i, component))
        for component in second_question_edge_components(edge_j, len(assembled[piece_j])):
            matched_edges.add((piece_j, component))
    total_error = 0.0
    max_piece_error = 0.0
    for piece_index, polygon in enumerate(assembled):
        best_error = None
        for edge_index, (edge_start, edge_end) in enumerate(polygon_edges(polygon)):
            if (piece_index, edge_index) in matched_edges:
                continue
            edge_vec = edge_end - edge_start
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-6:
                continue
            edge_unit = edge_vec / edge_len
            for side_start, side_end in rect_sides:
                side_vec = side_end - side_start
                side_len = float(np.linalg.norm(side_vec))
                if side_len <= 1e-6:
                    continue
                side_unit = side_vec / side_len
                parallel_error = 1.0 - abs(float(np.dot(edge_unit, side_unit)))
                start_error = point_rect_side_distance(edge_start, side_start, side_end)
                end_error = point_rect_side_distance(edge_end, side_start, side_end)
                error = (start_error + end_error) * 0.5 + parallel_error * side_len * 0.4
                if best_error is None or error < best_error:
                    best_error = error
        if best_error is None:
            best_error = 999.0
        max_piece_error = max(max_piece_error, best_error)
        total_error += best_error * best_error
    return total_error, max_piece_error


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted((max(0.0, float(start)), min(1.0, float(end))) for start, end in intervals)
    merged = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def second_question_matched_edge_components(assembled, matches):
    matched_edges = set()
    for match in matches:
        _, piece_i, edge_i, piece_j, edge_j = match[:5]
        for component in second_question_edge_components(edge_i, len(assembled[piece_i])):
            matched_edges.add((piece_i, component))
        for component in second_question_edge_components(edge_j, len(assembled[piece_j])):
            matched_edges.add((piece_j, component))
    return matched_edges


def second_question_boundary_cover_intervals(assembled, rect, matches):
    rect_points = order_points(cv2.boxPoints(rect))
    rect_sides = polygon_edges(rect_points)
    matched_edges = second_question_matched_edge_components(assembled, matches)
    side_intervals = [[] for _ in rect_sides]
    for piece_index, polygon in enumerate(assembled):
        for edge_index, (edge_start, edge_end) in enumerate(polygon_edges(polygon)):
            if (piece_index, edge_index) in matched_edges:
                continue
            edge_vec = edge_end - edge_start
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-6:
                continue
            edge_unit = edge_vec / edge_len
            edge_mid = (edge_start + edge_end) * 0.5
            for side_index, (side_start, side_end) in enumerate(rect_sides):
                side_vec = side_end - side_start
                side_len_sq = float(np.dot(side_vec, side_vec))
                side_len = math.sqrt(max(1e-6, side_len_sq))
                side_unit = side_vec / side_len
                if abs(float(np.dot(edge_unit, side_unit))) < SECOND_Q_BOUNDARY_COVER_PARALLEL_MIN:
                    continue
                distances = (
                    point_rect_side_distance(edge_start, side_start, side_end),
                    point_rect_side_distance(edge_mid, side_start, side_end),
                    point_rect_side_distance(edge_end, side_start, side_end),
                )
                if max(distances) > SECOND_Q_BOUNDARY_COVER_DISTANCE_PX:
                    continue
                t0 = float(np.dot(edge_start - side_start, side_vec) / side_len_sq)
                t1 = float(np.dot(edge_end - side_start, side_vec) / side_len_sq)
                side_intervals[side_index].append((min(t0, t1), max(t0, t1)))
    return rect_sides, side_intervals


def second_question_boundary_coverage_error(assembled, rect, matches):
    rect_sides, side_intervals = second_question_boundary_cover_intervals(assembled, rect, matches)
    total_error = 0.0
    max_missing_ratio = 0.0
    for intervals, (side_start, side_end) in zip(side_intervals, rect_sides):
        covered = sum(end - start for start, end in merge_intervals(intervals))
        missing_ratio = max(0.0, 1.0 - covered)
        side_len = float(np.linalg.norm(side_end - side_start))
        total_error += (missing_ratio * side_len) ** 2
        max_missing_ratio = max(max_missing_ratio, missing_ratio)
    return total_error, max_missing_ratio


def complement_intervals(intervals):
    merged = merge_intervals(intervals)
    result = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < 1.0:
        result.append((cursor, 1.0))
    return result


def second_question_boundary_repair_error(assembled, rect, matches):
    rect_sides, side_intervals = second_question_boundary_cover_intervals(assembled, rect, matches)
    matched_edges = second_question_matched_edge_components(assembled, matches)
    candidate_edges = []
    for piece_index, polygon in enumerate(assembled):
        for edge_index, (edge_start, edge_end) in enumerate(polygon_edges(polygon)):
            if (piece_index, edge_index) in matched_edges:
                continue
            edge_vec = edge_end - edge_start
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-6:
                continue
            candidate_edges.append((piece_index, edge_index, edge_start, edge_end, edge_vec / edge_len, edge_len))

    total_error = 0.0
    max_repairable_gap = 0.0
    repairable_count = 0
    for side_index, (side_start, side_end) in enumerate(rect_sides):
        side_vec = side_end - side_start
        side_len = float(np.linalg.norm(side_vec))
        if side_len <= 1e-6:
            continue
        side_unit = side_vec / side_len
        for start_t, end_t in complement_intervals(side_intervals[side_index]):
            gap_len = (end_t - start_t) * side_len
            if gap_len <= 1e-6:
                continue
            best_error = None
            for _piece_index, _edge_index, _edge_start, _edge_end, edge_unit, edge_len in candidate_edges:
                rel = abs(edge_len - gap_len) / max(edge_len, gap_len)
                if rel > SECOND_Q_BOUNDARY_REPAIR_REL_TOLERANCE:
                    continue
                # abs(dot) accepts both direct alignment and a 180-degree flipped placement.
                parallel = abs(float(np.dot(edge_unit, side_unit)))
                if parallel < SECOND_Q_BOUNDARY_COVER_PARALLEL_MIN:
                    continue
                error = rel * gap_len + (1.0 - parallel) * gap_len
                if best_error is None or error < best_error:
                    best_error = error
            if best_error is not None:
                repairable_count += 1
                max_repairable_gap = max(max_repairable_gap, gap_len)
                total_error += gap_len * gap_len + best_error * best_error
    return total_error, max_repairable_gap, repairable_count


def point_rect_side_distance(point, side_start, side_end):
    point = np.asarray(point, dtype=np.float32)
    side_start = np.asarray(side_start, dtype=np.float32)
    side_end = np.asarray(side_end, dtype=np.float32)
    side_vec = side_end - side_start
    side_len_sq = float(np.dot(side_vec, side_vec))
    if side_len_sq <= 1e-6:
        return float(np.linalg.norm(point - side_start))
    t = float(np.dot(point - side_start, side_vec) / side_len_sq)
    projection = side_start + side_vec * t
    line_distance = float(np.linalg.norm(point - projection))
    if t < 0.0:
        return float(np.linalg.norm(point - side_start))
    if t > 1.0:
        return float(np.linalg.norm(point - side_end))
    return line_distance


def second_question_edge_contact_error(assembled, rect, matches):
    rect_points = order_points(cv2.boxPoints(rect))
    rect_sides = polygon_edges(rect_points)
    internal_segments = []
    for match in matches:
        _, piece_i, edge_i, piece_j, edge_j, ia0, ia1, ja0, ja1 = match
        edge_i_start, edge_i_end = second_question_resolve_edge(assembled[piece_i], edge_i)
        edge_j_start, edge_j_end = second_question_resolve_edge(assembled[piece_j], edge_j)
        internal_segments.append((
            piece_i,
            set(second_question_edge_components(edge_i, len(assembled[piece_i]))),
            edge_i_start + (edge_i_end - edge_i_start) * ia0,
            edge_i_start + (edge_i_end - edge_i_start) * ia1,
        ))
        internal_segments.append((
            piece_j,
            set(second_question_edge_components(edge_j, len(assembled[piece_j]))),
            edge_j_start + (edge_j_end - edge_j_start) * ja0,
            edge_j_start + (edge_j_end - edge_j_start) * ja1,
        ))

    total_error = 0.0
    max_error = 0.0
    sample_positions = (0.25, 0.5, 0.75)
    for piece_index, polygon in enumerate(assembled):
        for edge_index, (edge_start, edge_end) in enumerate(polygon_edges(polygon)):
            for position in sample_positions:
                sample = edge_start + (edge_end - edge_start) * position
                best_distance = None
                for side_start, side_end in rect_sides:
                    dist = point_rect_side_distance(sample, side_start, side_end)
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                for owner_piece, owner_edges, seg_start, seg_end in internal_segments:
                    if owner_piece == piece_index and edge_index in owner_edges:
                        continue
                    dist = point_rect_side_distance(sample, seg_start, seg_end)
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                if best_distance is None:
                    best_distance = 999.0
                max_error = max(max_error, best_distance)
                total_error += best_distance * best_distance
    return total_error, max_error


def second_question_optimize_pose_graph(pieces, matches, initial):
    if len(pieces) < 3:
        return initial

    def pack(poses):
        values = []
        for transform in poses[1:]:
            values.extend([
                math.atan2(float(transform[1, 0]), float(transform[0, 0])),
                float(transform[0, 2]),
                float(transform[1, 2]),
            ])
        return np.asarray(values, dtype=float)

    def unpack(values):
        poses = [initial[0]]
        for index in range(len(pieces) - 1):
            theta, tx, ty = values[index * 3:index * 3 + 3]
            poses.append(rigid_transform(theta, tx, ty))
        return poses

    def residual(values):
        poses = unpack(values)
        result = []
        for match in matches:
            _, piece_i, _edge_i, piece_j, _edge_j = match[:5]
            ia, ib, ja, jb = second_question_match_segments(pieces, match)
            world_i = apply_homography_points(np.asarray([ia, ib], dtype=np.float32), poses[piece_i])
            world_j = apply_homography_points(np.asarray([jb, ja], dtype=np.float32), poses[piece_j])
            result.extend((world_i - world_j).ravel())
        return np.asarray(result, dtype=float)

    def cost(values):
        return float(np.linalg.norm(residual(values)))

    values = pack(initial)
    lam = 1e-3  # LM 阻尼:步长过大自动增大,收敛顺利则减小
    best_values, best_cost = values.copy(), cost(values)
    for _ in range(30):
        base_residual = residual(values)
        base_norm = float(np.linalg.norm(base_residual))
        if base_norm < 1e-9:
            break
        jacobian = np.empty((len(base_residual), len(values)))
        for index in range(len(values)):
            step = 1e-5 if index % 3 == 0 else 1e-3
            shifted = values.copy()
            shifted[index] += step
            jacobian[:, index] = (residual(shifted) - base_residual) / step
        try:
            delta = np.linalg.solve(
                jacobian.T @ jacobian + lam * np.eye(len(values)),
                -jacobian.T @ base_residual,
            )
        except np.linalg.LinAlgError:
            break
        candidate = values + delta
        if cost(candidate) < base_norm:
            values = candidate
            lam = max(lam * 0.5, 1e-6)
            if cost(candidate) < best_cost:
                best_values, best_cost = values.copy(), cost(candidate)
            if np.linalg.norm(delta) < 1e-7:
                break
        else:
            lam *= 5.0  # 发散则回退并加大阻尼
            if lam > 1e4:
                break
    if best_cost > cost(pack(initial)) + 1e-9:
        return initial  # 优化结果不如初始解,回退防发散
    return unpack(best_values)


def second_question_pairwise_overlap_area(assembled):
    overlap = 0.0
    for polygon_a, polygon_b in itertools.combinations(assembled, 2):
        contour_a = cv2.convexHull(np.asarray(polygon_a, dtype=np.float32).reshape(-1, 1, 2))
        contour_b = cv2.convexHull(np.asarray(polygon_b, dtype=np.float32).reshape(-1, 1, 2))
        try:
            area, _intersection = cv2.intersectConvexConvex(contour_a, contour_b)
            overlap += max(0.0, float(area))
        except Exception:
            pass
    return overlap


def second_question_fixed_rect_metrics():
    rect_w_px = float(SECOND_Q_FIXED_RECT_W_CM) * (WARP_W - 1) / A4_W_CM
    rect_h_px = float(SECOND_Q_FIXED_RECT_H_CM) * (WARP_H - 1) / A4_H_CM
    area = max(1.0, rect_w_px * rect_h_px)
    aspect = max(rect_w_px, rect_h_px) / max(1.0, min(rect_w_px, rect_h_px))
    perimeter = 2.0 * (rect_w_px + rect_h_px)
    return area, aspect, perimeter


def polygon_internal_angle(prev_point, point, next_point):
    v1 = np.asarray(prev_point, dtype=np.float32) - np.asarray(point, dtype=np.float32)
    v2 = np.asarray(next_point, dtype=np.float32) - np.asarray(point, dtype=np.float32)
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 <= 1e-6 or len2 <= 1e-6:
        return 180.0
    cosine = max(-1.0, min(1.0, float(np.dot(v1, v2) / (len1 * len2))))
    return math.degrees(math.acos(cosine))


def second_question_right_angle_points(assembled):
    points = []
    for polygon in assembled:
        polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        count = len(polygon)
        for index, point in enumerate(polygon):
            angle = polygon_internal_angle(polygon[index - 1], point, polygon[(index + 1) % count])
            if SECOND_Q_RIGHT_ANGLE_MIN_DEG <= angle <= SECOND_Q_RIGHT_ANGLE_MAX_DEG:
                points.append(point)
    return points


def second_question_right_corner_error(assembled, rect):
    rect_points = order_points(cv2.boxPoints(rect)).astype(np.float32)
    right_points = second_question_right_angle_points(assembled)
    if not right_points:
        return (
            float(len(rect_points)) * SECOND_Q_RIGHT_CORNER_MISSING_PX ** 2,
            SECOND_Q_RIGHT_CORNER_MISSING_PX,
            0,
        )

    used = set()
    total_error = 0.0
    max_error = 0.0
    for corner in rect_points:
        best_index = None
        best_distance = None
        for index, point in enumerate(right_points):
            if index in used:
                continue
            distance = float(np.linalg.norm(np.asarray(point, dtype=np.float32) - corner))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            best_distance = SECOND_Q_RIGHT_CORNER_MISSING_PX
        else:
            used.add(best_index)
        total_error += best_distance * best_distance
        max_error = max(max_error, best_distance)
    return total_error, max_error, len(right_points)


def second_question_score_assembly_fast(
    transforms,
    assembled,
    matches,
    closure_error,
    reject_stats=None,
    apply_rectlike_reject=True,
    use_fixed_rect_score=None,
    coarse_score=False,
):
    if use_fixed_rect_score is None:
        use_fixed_rect_score = SECOND_Q_USE_FIXED_RECT_SCORE
    all_points = np.vstack(assembled).astype(np.float32)
    min_point = all_points.min(axis=0)
    max_point = all_points.max(axis=0)
    width, height = np.ceil(max_point - min_point + SECOND_Q_SOLVE_PADDING_PX * 2).astype(int)
    if width <= 1 or height <= 1 or width > WARP_W * 3 or height > WARP_H * 3:
        return second_question_reject(reject_stats, "bad_canvas")

    rect = cv2.minAreaRect(all_points)
    rect_w, rect_h = rect[1]
    rect_area = float(rect_w * rect_h)
    if rect_area <= 1e-6:
        return second_question_reject(reject_stats, "bad_rect")
    if (apply_rectlike_reject or coarse_score) and not second_question_dimensions_in_range(rect_w, rect_h):
        return second_question_reject(reject_stats, "矩形_尺寸越界")

    piece_area = sum(
        abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
        for polygon in assembled
    )
    piece_area = max(1.0, piece_area)
    fixed_area, fixed_aspect, fixed_perimeter = second_question_fixed_rect_metrics()
    expected_area = fixed_area if use_fixed_rect_score else piece_area
    overlap = second_question_pairwise_overlap_area(assembled)
    union_area = max(1.0, piece_area - overlap)
    fill_error = max(0.0, rect_area - union_area)
    area_gap = max(0.0, rect_area - expected_area)
    area_gap_ratio = area_gap / expected_area
    aspect = max(float(rect_w), float(rect_h)) / max(1.0, min(float(rect_w), float(rect_h)))
    aspect_error = abs(math.log(max(aspect, 1e-6) / max(fixed_aspect, 1e-6))) if use_fixed_rect_score else 0.0
    hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    perimeter = float(cv2.arcLength(hull, True))
    perimeter_error = abs(perimeter - fixed_perimeter) if use_fixed_rect_score else 0.0
    hull_gap = max(0.0, hull_area - expected_area)
    hull_gap_ratio = hull_gap / expected_area
    overlap_ratio = overlap / union_area
    if len(assembled) == 2 and overlap_ratio > SECOND_Q_TWO_PIECE_MAX_OVERLAP_RATIO:
        return second_question_reject(reject_stats, "overlap")
    if len(assembled) == 2 and hull_gap_ratio > SECOND_Q_TWO_PIECE_MAX_HULL_GAP_RATIO:
        return second_question_reject(reject_stats, "hull")

    if coarse_score:
        boundary_error = 0.0
        boundary_max_error = 0.0
        boundary_cover_error = 0.0
        boundary_missing_ratio = 0.0
    else:
        boundary_error, boundary_max_error = second_question_boundary_edge_error(
            assembled, rect, matches)
        boundary_cover_error, boundary_missing_ratio = second_question_boundary_coverage_error(
            assembled, rect, matches)
        if SECOND_Q_STRICT_BOUNDARY_REJECT and (
            boundary_max_error > SECOND_Q_BOUNDARY_EDGE_MAX_ERROR_PX
            or boundary_missing_ratio > 0.20
        ):
            return second_question_reject(reject_stats, "矩形_边界不完整")

    # fast 路径只保留外轮廓近似评分；默认矩形硬约束开启时不会走到这里。
    match_error = sum(float(match[0]) for match in matches) * 3000.0
    score = (
        closure_error * 400.0
        + overlap * 12.0
        + fill_error * 8.0
        + abs(union_area - expected_area) * (SECOND_Q_FIXED_AREA_WEIGHT if use_fixed_rect_score else 4.0)
        + abs(rect_area - expected_area) * 3.0
        + aspect_error * SECOND_Q_FIXED_ASPECT_WEIGHT
        + perimeter_error * SECOND_Q_FIXED_PERIMETER_WEIGHT
        + boundary_error * SECOND_Q_BOUNDARY_EDGE_WEIGHT
        + boundary_cover_error * SECOND_Q_BOUNDARY_COVER_WEIGHT
        + match_error
    )
    detail = {
        "overlap": overlap,
        "fill": union_area / rect_area,
        "aspect": aspect,
        "area_gap": area_gap,
        "area_gap_ratio": area_gap_ratio,
        "hull_gap": hull_gap,
        "hull_gap_ratio": hull_gap_ratio,
        "boundary": boundary_error,
        "boundary_cover": boundary_missing_ratio,
        "boundary_max": boundary_max_error,
        "boundary_cover_error": boundary_cover_error,
        "dynamic_rect": not bool(use_fixed_rect_score),
        "boundary_repair": 0.0,
        "boundary_repair_count": 0,
        "edge_contact": 0.0,
        "edge_contact_max": 0.0,
        "right_corner": 0.0,
        "right_corner_count": 0,
        "rect_w": float(rect_w),
        "rect_h": float(rect_h),
        "union_area": union_area,
        "rect_area": rect_area,
        "fixed_area": fixed_area,
        "aspect_error": aspect_error,
        "perimeter_error": perimeter_error,
        "match_error": match_error,
        "closure": closure_error,
    }
    return score, transforms, assembled, detail


def second_question_score_assembly(
    transforms,
    assembled,
    matches,
    closure_error,
    reject_stats=None,
    apply_rectlike_reject=True,
    use_fixed_rect_score=None,
    mask_scale=0.5,
    coarse_score=False,
):
    if use_fixed_rect_score is None:
        use_fixed_rect_score = SECOND_Q_USE_FIXED_RECT_SCORE
    if SECOND_Q_USE_FAST_GEOMETRY_SCORE and not SECOND_Q_RECTLIKE_REJECT_ENABLED:
        return second_question_score_assembly_fast(
            transforms, assembled, matches, closure_error, reject_stats,
            apply_rectlike_reject=apply_rectlike_reject,
            use_fixed_rect_score=use_fixed_rect_score,
        )

    all_points = np.vstack(assembled)
    min_point = all_points.min(axis=0)
    max_point = all_points.max(axis=0)
    # 最终评分按 0.5 倍降采样；auto 全量粗筛使用更小掩膜，只负责候选排序。
    # 面积和长度按比例还原，粗筛结果不会直接作为最终机械目标。
    scale = max(0.08, float(mask_scale))
    inv_area = 1.0 / (scale * scale)
    inv_len = 1.0 / scale
    shift = -min_point * scale + SECOND_Q_SOLVE_PADDING_PX
    width, height = np.ceil((max_point - min_point) * scale + SECOND_Q_SOLVE_PADDING_PX * 2).astype(int)
    if width <= 1 or height <= 1 or width > WARP_W * 3 or height > WARP_H * 3:
        return second_question_reject(reject_stats, "bad_canvas")

    total = np.zeros((int(height), int(width)), dtype=np.uint8)
    for polygon in assembled:
        mask = np.zeros_like(total)
        cv2.fillPoly(mask, [np.rint(polygon * scale + shift).astype(np.int32)], 1)
        total = total + mask
    overlap = float(np.count_nonzero(total > 1)) * inv_area
    union = (total > 0).astype(np.uint8)
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return second_question_reject(reject_stats, "no_contour")
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    contour_world = (contour_points - shift.reshape(1, 2)) * inv_len
    rect = cv2.minAreaRect(contour_world)
    rect_w, rect_h = rect[1]
    rect_area = float(rect_w * rect_h)
    union_area = float(np.count_nonzero(union)) * inv_area
    if rect_area <= 1e-6:
        return second_question_reject(reject_stats, "bad_rect")
    if union_area <= 1e-6:
        return second_question_reject(reject_stats, "empty_union")

    piece_area = sum(
        abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
        for polygon in assembled
    )
    piece_area = max(1.0, piece_area)
    fixed_area, fixed_aspect, fixed_perimeter = second_question_fixed_rect_metrics()
    expected_area = fixed_area if use_fixed_rect_score else piece_area
    fill_error = max(0.0, rect_area - union_area)
    area_gap = max(0.0, rect_area - expected_area)
    area_gap_ratio = area_gap / expected_area
    aspect = max(float(rect_w), float(rect_h)) / max(1.0, min(float(rect_w), float(rect_h)))
    aspect_error = abs(math.log(max(aspect, 1e-6) / max(fixed_aspect, 1e-6))) if use_fixed_rect_score else 0.0
    disconnected_area = float(sum(cv2.contourArea(item) for item in contours) - cv2.contourArea(contour)) * inv_area
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)) * inv_area)
    perimeter = float(cv2.arcLength(contour, True)) * inv_len
    perimeter_error = abs(perimeter - fixed_perimeter) if use_fixed_rect_score else 0.0
    hull_gap = max(0.0, hull_area - float(cv2.contourArea(contour)) * inv_area)
    hull_gap_ratio = hull_gap / expected_area
    overlap_ratio = overlap / union_area
    if len(assembled) == 2 and overlap_ratio > SECOND_Q_TWO_PIECE_MAX_OVERLAP_RATIO:
        return second_question_reject(reject_stats, "overlap")
    if len(assembled) == 2 and hull_gap_ratio > SECOND_Q_TWO_PIECE_MAX_HULL_GAP_RATIO:
        return second_question_reject(reject_stats, "hull")

    if (apply_rectlike_reject or coarse_score) and not second_question_dimensions_in_range(rect_w, rect_h):
        return second_question_reject(reject_stats, "矩形_尺寸越界")

    if coarse_score:
        # 边界覆盖包含多层 Python 循环，是板端全量搜索的主要热点。
        # 粗筛只按尺寸/填充/重叠排序，完整边界验收留给最终候选。
        boundary_error = 0.0
        boundary_max_error = 0.0
        boundary_cover_error = 0.0
        boundary_missing_ratio = 0.0
    else:
        boundary_error, boundary_max_error = second_question_boundary_edge_error(
            assembled, rect, matches)
        boundary_cover_error, boundary_missing_ratio = second_question_boundary_coverage_error(
            assembled, rect, matches)
        if SECOND_Q_STRICT_BOUNDARY_REJECT and (
            boundary_max_error > SECOND_Q_BOUNDARY_EDGE_MAX_ERROR_PX
            or boundary_missing_ratio > 0.20
        ):
            return second_question_reject(reject_stats, "矩形_边界不完整")

    # 低分辨率轮廓会产生量化折角，粗筛不能据此硬拒绝；最终评分仍执行完整矩形约束。
    rectlike_reason = second_question_rectlike_reject_reason(
        contour_world,
        rect_area,
        union_area,
        hull_area,
        aspect,
        fixed_aspect,
    ) if apply_rectlike_reject and not coarse_score else None
    if rectlike_reason is not None:
        return second_question_reject(reject_stats, rectlike_reason)

    # 评分项与参考实现对齐:闭环/重叠/填充/面积/长宽比/周长/匹配惩罚。
    match_error = sum(float(match[0]) for match in matches) * 3000.0
    score = (
        closure_error * 400.0
        + overlap * 12.0
        + fill_error * 8.0
        + abs(union_area - expected_area) * (SECOND_Q_FIXED_AREA_WEIGHT if use_fixed_rect_score else 4.0)
        + abs(rect_area - expected_area) * 3.0
        + aspect_error * SECOND_Q_FIXED_ASPECT_WEIGHT
        + perimeter_error * SECOND_Q_FIXED_PERIMETER_WEIGHT
        + disconnected_area * 20.0
        + boundary_error * SECOND_Q_BOUNDARY_EDGE_WEIGHT
        + boundary_cover_error * SECOND_Q_BOUNDARY_COVER_WEIGHT
        + match_error
    )
    detail = {
        "overlap": overlap,
        "fill": union_area / rect_area,
        "aspect": aspect,
        "area_gap": area_gap,
        "area_gap_ratio": area_gap_ratio,
        "hull_gap": hull_gap,
        "hull_gap_ratio": hull_gap_ratio,
        "boundary": boundary_error,
        "boundary_cover": boundary_missing_ratio,
        "boundary_max": boundary_max_error,
        "boundary_cover_error": boundary_cover_error,
        "dynamic_rect": not bool(use_fixed_rect_score),
        "boundary_repair": 0.0,
        "boundary_repair_count": 0,
        "edge_contact": 0.0,
        "edge_contact_max": 0.0,
        "right_corner": 0.0,
        "right_corner_count": 0,
        "rect_w": float(rect_w),
        "rect_h": float(rect_h),
        "union_area": union_area,
        "rect_area": rect_area,
        "fixed_area": fixed_area,
        "aspect_error": aspect_error,
        "perimeter_error": perimeter_error,
        "match_error": match_error,
        "closure": closure_error,
    }
    return score, transforms, assembled, detail


def second_question_dimensions_in_range(rect_w, rect_h):
    dims_px = sorted([float(rect_w), float(rect_h)], reverse=True)
    long_cm = dims_px[0] * A4_W_CM / max(1.0, WARP_W - 1)
    short_cm = dims_px[1] * A4_H_CM / max(1.0, WARP_H - 1)
    min_long = max(SECOND_Q_RECT_MIN_W_CM, SECOND_Q_RECT_MIN_H_CM) - SECOND_Q_DIMENSION_TOLERANCE_CM
    max_long = max(SECOND_Q_RECT_MAX_W_CM, SECOND_Q_RECT_MAX_H_CM) + SECOND_Q_DIMENSION_TOLERANCE_CM
    min_short = min(SECOND_Q_RECT_MIN_W_CM, SECOND_Q_RECT_MIN_H_CM) - SECOND_Q_DIMENSION_TOLERANCE_CM
    max_short = min(SECOND_Q_RECT_MAX_W_CM, SECOND_Q_RECT_MAX_H_CM) + SECOND_Q_DIMENSION_TOLERANCE_CM
    return min_long <= long_cm <= max_long and min_short <= short_cm <= max_short


def second_question_dimension_range_penalty(rect_w, rect_h):
    dims_px = sorted([float(rect_w), float(rect_h)], reverse=True)
    dims_cm = [
        dims_px[0] * A4_W_CM / max(1.0, WARP_W - 1),
        dims_px[1] * A4_H_CM / max(1.0, WARP_H - 1),
    ]
    max_dim = max(SECOND_Q_RECT_MAX_W_CM, SECOND_Q_RECT_MAX_H_CM)
    min_dim = min(SECOND_Q_RECT_MIN_W_CM, SECOND_Q_RECT_MIN_H_CM)
    penalty = 0.0
    if dims_cm[0] > max_dim:
        penalty += dims_cm[0] - max_dim
    if dims_cm[1] < min_dim:
        penalty += min_dim - dims_cm[1]
    return penalty


def second_question_match_summary(matches):
    parts = []
    for match in matches:
        _, piece_i, edge_i, piece_j, edge_j = match[:5]
        parts.append(
            "P%d%s-P%d%s"
            % (
                piece_i,
                second_question_edge_label(edge_i),
                piece_j,
                second_question_edge_label(edge_j),
            )
        )
    return ",".join(parts)


def second_question_candidate_kind(match):
    if tuple(match[5:]) == (0.0, 1.0, 0.0, 1.0):
        return "full"
    return "partial[%.2f-%.2f|%.2f-%.2f]" % tuple(float(value) for value in match[5:])


def second_question_print_piece_edges(pieces):
    for piece_index, polygon in enumerate(pieces):
        contour_area = abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
        center = np.mean(np.asarray(polygon, dtype=np.float32).reshape(-1, 2), axis=0)
        edge_parts = []
        min_edge = None
        for edge_index, (edge_start, edge_end) in enumerate(polygon_edges(polygon)):
            edge_len = float(np.linalg.norm(edge_end - edge_start))
            min_edge = edge_len if min_edge is None else min(min_edge, edge_len)
            edge_parts.append("E%d=%.1f" % (edge_index, edge_len))
        print(
            "P%d points=%d area=%.0f center=(%.1f,%.1f) %s"
            % (
                piece_index,
                len(polygon),
                contour_area,
                float(center[0]),
                float(center[1]),
                " ".join(edge_parts),
            )
        )
        if min_edge is not None and min_edge < SECOND_Q_DEBUG_SHORT_EDGE_PX:
            print(
                "WARN_SHORT_EDGE P%d min=%.1f threshold=%.1f"
                % (piece_index, float(min_edge), float(SECOND_Q_DEBUG_SHORT_EDGE_PX))
            )


def second_question_print_candidate_edges(pieces, candidates):
    pair_candidates = {}
    for candidate in candidates:
        pair = tuple(sorted((candidate[1], candidate[3])))
        pair_candidates.setdefault(pair, []).append(candidate)

    for piece_i, piece_j in itertools.combinations(range(len(pieces)), 2):
        pair = (piece_i, piece_j)
        matches = pair_candidates.get(pair, [])
        if not matches:
            print("无候选 P%d-P%d" % (piece_i, piece_j))
            continue
        for match in matches[:SECOND_Q_DEBUG_MAX_CAND_PRINT]:
            score, mi, edge_i, mj, edge_j = match[:5]
            print(
                "CAND P%d%s-P%d%s rel=%.3f %s"
                % (
                    mi,
                    second_question_edge_label(edge_i),
                    mj,
                    second_question_edge_label(edge_j),
                    float(score),
                    second_question_candidate_kind(match),
                )
            )
        if len(matches) > SECOND_Q_DEBUG_MAX_CAND_PRINT:
            print(
                "CAND_MORE P%d-P%d omitted=%d"
                % (piece_i, piece_j, len(matches) - SECOND_Q_DEBUG_MAX_CAND_PRINT)
            )


def second_question_print_top_candidates(top_candidates):
    for rank, candidate in enumerate(top_candidates[:SECOND_Q_DEBUG_TOP_N], start=1):
        score, cut_mode, matches, detail = candidate
        print(
            "SECOND Q TOP%d mode=%s score=%.1f fill=%.3f gap=%.3f hull=%.3f aspect=%.3f fix=%.0f/%.3f/%.0f overlap=%.0f boundary=%.1f cover=%.2f repair=%.1f/%d edge=%.1f right=%.1f/%d rect=%.0fx%.0f m=%s"
            % (
                rank,
                cut_mode,
                float(score),
                float(detail.get("fill", 0.0)),
                float(detail.get("area_gap_ratio", 0.0)),
                float(detail.get("hull_gap_ratio", 0.0)),
                float(detail.get("aspect", 0.0)),
                abs(float(detail.get("rect_area", 0.0)) - float(detail.get("fixed_area", 0.0))),
                float(detail.get("aspect_error", 0.0)),
                float(detail.get("perimeter_error", 0.0)),
                float(detail.get("overlap", 0.0)),
                float(detail.get("boundary", 0.0)),
                float(detail.get("boundary_cover", 0.0)),
                float(detail.get("boundary_repair", 0.0)),
                int(detail.get("boundary_repair_count", 0)),
                float(detail.get("edge_contact", 0.0)),
                float(detail.get("right_corner", 0.0)),
                int(detail.get("right_corner_count", 0)),
                float(detail.get("rect_w", 0.0)),
                float(detail.get("rect_h", 0.0)),
                second_question_match_summary(matches),
            )
        )


def second_question_congruent_rect_dimensions(pieces):
    """全部碎片是彼此全等的矩形时返回测得的 (长边, 短边),否则 None。

    判定:4 顶点、各内角 90°±10°、对边相等(±4%)、片间尺寸一致(±4%)。
    用于等分矩形/条带的槽位快车道,避免逐组合评分。
    """
    if not pieces or any(len(piece) != 4 for piece in pieces):
        return None
    dims = []
    for piece in pieces:
        angles = second_question_polygon_angles(piece)
        if np.any(np.abs(angles - math.pi / 2) > math.radians(10.0)):
            return None
        lengths = [
            float(np.linalg.norm(edge_end - edge_start))
            for edge_start, edge_end in polygon_edges(piece)
        ]
        if abs(lengths[0] - lengths[2]) > 0.04 * max(lengths[0], lengths[2]):
            return None
        if abs(lengths[1] - lengths[3]) > 0.04 * max(lengths[1], lengths[3]):
            return None
        dims.append(sorted([
            (lengths[0] + lengths[2]) / 2,
            (lengths[1] + lengths[3]) / 2,
        ], reverse=True))
    first = np.asarray(dims[0], dtype=float)
    for dim in dims[1:]:
        if np.max(np.abs(np.asarray(dim, dtype=float) - first) / first) > 0.04:
            return None
    return float(first[0]), float(first[1])


def second_question_first_template_is_confident(match):
    return bool(
        match is not None
        and float(match.get("cost", 999.0)) <= SECOND_Q_FIRST_TEMPLATE_CONFIDENT_MAX_COST
        and float(match.get("max_shape_score", 999.0)) <= SECOND_Q_FIRST_TEMPLATE_CONFIDENT_MAX_SHAPE
    )


def second_question_first_template_allowed_pairs(match):
    if (
        not SECOND_Q_FIRST_TEMPLATE_TOPOLOGY_FIRST
        or match is None
        or len(match.get("assignment", ())) != len(FIRST_Q_TEMPLATES)
    ):
        return None

    # 第一问官方切法里的内部相邻关系。这里只约束“哪些碎片可以相邻”,
    # 具体哪条边怎么贴仍由第二问候选边匹配和矩形评分决定。
    official_pairs = {
        ("A", "B"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    }
    template_to_piece = {}
    for piece_index, template_index in enumerate(match["assignment"]):
        template_to_piece[FIRST_Q_TEMPLATES[template_index]["name"]] = piece_index

    pairs = set()
    for name_a, name_b in official_pairs:
        if name_a in template_to_piece and name_b in template_to_piece:
            pairs.add(tuple(sorted((template_to_piece[name_a], template_to_piece[name_b]))))
    return pairs or None


def second_question_first_template_ring_priorities(pieces, candidates, match):
    """返回官方 A-B-C-D 外圈闭环及按实测接缝长度计算的候选优先级。

    模板只用于决定先搜索哪些相邻关系和预期接缝长度；候选位姿、最终轮廓
    和机械坐标仍完全来自实测多边形，不使用模板坐标直接摆放。
    """
    if match is None or len(match.get("assignment", ())) != 4:
        return None, None
    template_to_piece = {}
    piece_names = {}
    for piece_index, template_index in enumerate(match["assignment"]):
        name = FIRST_Q_TEMPLATES[template_index]["name"]
        template_to_piece[name] = piece_index
        piece_names[piece_index] = name
    seam_lengths_cm = {
        frozenset(("A", "B")): 3.6878,
        frozenset(("B", "C")): 7.6942,
        frozenset(("C", "D")): 3.0,
        frozenset(("D", "A")): 2.0,
    }
    pair_lengths = {}
    for names, length_cm in seam_lengths_cm.items():
        name_a, name_b = tuple(names)
        if name_a not in template_to_piece or name_b not in template_to_piece:
            return None, None
        pair = tuple(sorted((template_to_piece[name_a], template_to_piece[name_b])))
        pair_lengths[pair] = float(length_cm)

    measured_area = sum(
        abs(float(cv2.contourArea(np.asarray(piece, dtype=np.float32))))
        for piece in pieces
    )
    px_per_cm = math.sqrt(max(1.0, measured_area) / 60.0)
    priorities = {}
    for candidate in candidates:
        pair = tuple(sorted((candidate[1], candidate[3])))
        expected_cm = pair_lengths.get(pair)
        if expected_cm is None:
            continue
        ia, ib, ja, jb = second_question_match_segments(pieces, candidate)
        measured_length = 0.5 * (
            float(np.linalg.norm(ib - ia)) + float(np.linalg.norm(jb - ja))
        )
        expected_length = max(1.0, expected_cm * px_per_cm)
        length_error = abs(measured_length - expected_length) / expected_length
        _, piece_i, _edge_i, piece_j, _edge_j, ia0, ia1, ja0, ja1 = candidate
        interval_by_piece = {piece_i: (ia0, ia1), piece_j: (ja0, ja1)}
        structure_error = 0.0
        for piece_index in pair:
            start_t, end_t = interval_by_piece[piece_index]
            if piece_names[piece_index] == "D":
                # D 的完整斜边约 10 cm，外圈接缝只占靠端点的一段。
                structure_error += abs((end_t - start_t) - expected_cm / 10.0)
            else:
                structure_error += abs(start_t) + abs(1.0 - end_t)
        priorities[candidate] = (
            length_error
            + 0.35 * structure_error
            + 0.05 * float(candidate[0])
        )
    return set(pair_lengths), priorities


def second_question_template_vertex_correspondences(piece, template_index, limit=None):
    """返回按误差排序的模板顶点映射，允许实测轮廓多出共线/短边顶点。"""
    template_cm = np.asarray(FIRST_Q_TEMPLATES[template_index]["polygon_cm"], dtype=np.float32)
    template = np.empty_like(template_cm)
    template[:, 0] = template_cm[:, 0] * (WARP_W - 1) / A4_W_CM
    template[:, 1] = template_cm[:, 1] * (WARP_H - 1) / A4_H_CM
    perimeter = float(cv2.arcLength(template.reshape(-1, 1, 2), True))
    template = cv2.approxPolyDP(
        template.reshape(-1, 1, 2), max(1e-4, perimeter * 0.001), True
    ).reshape(-1, 2).astype(np.float32)
    template = second_question_normalize_polygon_winding(template)
    measured = second_question_normalize_polygon_winding(piece)
    if len(template) < 3 or len(measured) < len(template):
        return []

    target = template - np.mean(template, axis=0)
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 1e-6:
        return []
    target /= target_norm
    ranked = {}
    for selected in itertools.combinations(range(len(measured)), len(template)):
        base_indices = np.asarray(selected, dtype=np.int32)
        for ordered_indices in (base_indices, base_indices[::-1]):
            for shift in range(len(template)):
                indices = np.roll(ordered_indices, -shift)
                source = measured[indices] - np.mean(measured[indices], axis=0)
                source_norm = float(np.linalg.norm(source))
                if source_norm <= 1e-6:
                    continue
                source /= source_norm
                angle = rotation_angle_between_point_sets(source, target)
                mapped = rotate_points_clockwise(source, angle)
                error = float(np.mean(np.sum((mapped - target) ** 2, axis=1)))
                key = tuple(int(index) for index in indices)
                if key not in ranked or error < ranked[key]:
                    ranked[key] = error
    candidates = [
        (error, np.asarray(indices, dtype=np.int32))
        for indices, error in ranked.items()
    ]
    candidates.sort(key=lambda item: item[0])
    if limit is not None:
        candidates = candidates[:max(0, int(limit))]
    return candidates


def second_question_template_vertex_correspondence(piece, template_index):
    """兼容旧调用：返回误差最低的一套模板顶点映射。"""
    candidates = second_question_template_vertex_correspondences(
        piece, template_index, limit=1)
    return None if not candidates else candidates[0][1]


def second_question_template_edge_ref(
    vertex_map,
    template_edge_index,
    measured_point_count,
    interval=(0.0, 1.0),
):
    """把模板边和模板方向区间换算为实测多边形边编号及区间。"""
    template_count = len(vertex_map)
    measured_point_count = int(measured_point_count)
    start_index = int(vertex_map[template_edge_index % template_count])
    end_index = int(vertex_map[(template_edge_index + 1) % template_count])
    start_t, end_t = float(interval[0]), float(interval[1])
    forward_span = (end_index - start_index) % measured_point_count
    reverse_span = (start_index - end_index) % measured_point_count
    if 1 <= forward_span <= SECOND_Q_COMPOSITE_EDGE_MAX_SPAN:
        edge_ref = start_index if forward_span == 1 else (start_index, forward_span)
        return edge_ref, start_t, end_t
    if 1 <= reverse_span <= SECOND_Q_COMPOSITE_EDGE_MAX_SPAN:
        edge_ref = end_index if reverse_span == 1 else (end_index, reverse_span)
        return edge_ref, 1.0 - end_t, 1.0 - start_t
    return None


def second_question_first_template_guided_match_sets(pieces, match):
    """由多套顶点映射生成并排序官方第一问的引导接缝组合。"""
    if match is None or len(match.get("assignment", ())) != 4:
        return ()
    name_to_piece = {}
    map_candidates = {}
    for piece_index, template_index in enumerate(match["assignment"]):
        name = FIRST_Q_TEMPLATES[template_index]["name"]
        name_to_piece[name] = piece_index
        candidates = second_question_template_vertex_correspondences(
            pieces[piece_index],
            template_index,
            limit=SECOND_Q_FIRST_TEMPLATE_GUIDED_MAPS_PER_PIECE,
        )
        if not candidates:
            return ()
        map_candidates[name] = candidates

    # D 简化后是三角形，其模板边 2 是 10 cm 长斜边。A、C 分别接两端。
    seam_specs = (
        ("A", 2, (0.0, 1.0), "B", 0, (0.0, 1.0)),
        ("B", 2, (0.0, 1.0), "C", 0, (0.0, 1.0)),
        ("C", 1, (0.0, 1.0), "D", 2, (0.0, 0.30)),
        ("D", 2, (0.80, 1.0), "A", 1, (0.0, 1.0)),
    )
    names = ("A", "B", "C", "D")
    ranked_sets = {}
    candidate_lists = [map_candidates[name] for name in names]
    for selected_maps in itertools.product(*candidate_lists):
        mapping_error = sum(float(item[0]) for item in selected_maps)
        vertex_maps = {
            name: selected_maps[index][1] for index, name in enumerate(names)
        }
        matches = []
        valid = True
        for name_i, edge_i, interval_i, name_j, edge_j, interval_j in seam_specs:
            ref_i = second_question_template_edge_ref(
                vertex_maps[name_i], edge_i,
                len(pieces[name_to_piece[name_i]]), interval_i)
            ref_j = second_question_template_edge_ref(
                vertex_maps[name_j], edge_j,
                len(pieces[name_to_piece[name_j]]), interval_j)
            if ref_i is None or ref_j is None:
                valid = False
                break
            measured_edge_i, ia0, ia1 = ref_i
            measured_edge_j, ja0, ja1 = ref_j
            piece_i = name_to_piece[name_i]
            piece_j = name_to_piece[name_j]
            a, b, c, d = second_question_match_segments(pieces, (
                0.0, piece_i, measured_edge_i, piece_j, measured_edge_j,
                ia0, ia1, ja0, ja1,
            ))
            length_i = float(np.linalg.norm(b - a))
            length_j = float(np.linalg.norm(d - c))
            penalty = abs(length_i - length_j) / max(1.0, max(length_i, length_j))
            matches.append((
                penalty, piece_i, measured_edge_i, piece_j, measured_edge_j,
                ia0, ia1, ja0, ja1,
            ))
        if not valid:
            continue
        match_set = tuple(matches)
        signature = tuple(item[1:] for item in match_set)
        rank = mapping_error + sum(float(item[0]) for item in match_set)
        previous = ranked_sets.get(signature)
        if previous is None or rank < previous[0]:
            ranked_sets[signature] = (rank, match_set)
    ordered = sorted(ranked_sets.values(), key=lambda item: item[0])
    return tuple(
        match_set for _rank, match_set in
        ordered[:SECOND_Q_FIRST_TEMPLATE_GUIDED_MAX_SETS]
    )


def second_question_first_template_guided_matches(pieces, match):
    """兼容旧调用：返回排名第一的官方引导接缝组合。"""
    match_sets = second_question_first_template_guided_match_sets(pieces, match)
    return None if not match_sets else match_sets[0]


def second_question_congruent_rect_transforms(pieces):
    """全等矩形碎片直接摆放进目标矩形槽位,不匹配时返回 None。

    布局(2x2 或条带)按实测尺寸选择;完全相同的空白矩形没有可观测身份,
    检测片与目标槽位的任意双射都是正确解。
    """
    count = len(pieces)
    dims = second_question_congruent_rect_dimensions(pieces)
    if dims is None:
        return None

    long_side, short_side = float(dims[0]), float(dims[1])
    if count == 4:
        grids = ((2, 2), (4, 1), (1, 4))
    else:
        grids = ((count, 1), (1, count))

    fixed_w_px = float(SECOND_Q_FIXED_RECT_W_CM) * (WARP_W - 1) / A4_W_CM
    fixed_h_px = float(SECOND_Q_FIXED_RECT_H_CM) * (WARP_H - 1) / A4_H_CM
    fixed_dims = sorted((fixed_w_px, fixed_h_px), reverse=True)
    layouts = []
    for columns, rows in grids:
        for cell_w, cell_h in ((long_side, short_side), (short_side, long_side)):
            target_w = float(columns) * cell_w
            target_h = float(rows) * cell_h
            if not second_question_dimensions_in_range(target_w, target_h):
                continue
            slots = [
                (float(column) * cell_w, float(row) * cell_h)
                for row in range(rows)
                for column in range(columns)
            ]
            target_dims = sorted((target_w, target_h), reverse=True)
            preference = (
                abs(target_dims[0] - fixed_dims[0])
                + abs(target_dims[1] - fixed_dims[1])
            )
            layouts.append((preference, cell_w, cell_h, slots))
    if not layouts:
        return None
    _preference, cell_w, cell_h, slots = min(layouts, key=lambda item: item[0])
    transforms = []
    for piece, slot in zip(pieces, slots):
        best = None
        for edge_start, edge_end in polygon_edges(piece):
            vector = edge_end - edge_start
            angle = -math.atan2(float(vector[1]), float(vector[0]))
            rotation = rigid_transform(angle, 0.0, 0.0)
            rotated = apply_homography_points(piece, rotation)
            low, high = rotated.min(axis=0), rotated.max(axis=0)
            size = high - low
            cost = abs(size[0] - cell_w) + abs(size[1] - cell_h)
            if best is None or cost < best[0]:
                best = (cost, rotation, low)
        _, rotation, low = best
        translation = rigid_transform(
            0.0, float(slot[0] - low[0]), float(slot[1] - low[1]))
        transforms.append(translation.dot(rotation))
    return transforms


def second_question_normalize_polygon_winding(polygon):
    """统一多边形绕向，避免混合顺/逆时针导致拼接落到接缝同一侧。"""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) >= 3 and cv2.contourArea(points, oriented=True) < 0.0:
        points = points[::-1].copy()
    return points


# 第二问求解主入口：输入检测到的碎片多边形，输出每块碎片的搬运变换、
# 归一化矩形位置和边匹配结果。后续 attach_second_question_targets 会把它转成移动计划。
def second_question_solve_transforms(piece_polygons, first_template_match=None):
    pieces = [second_question_normalize_polygon_winding(polygon) for polygon in piece_polygons]
    if not 2 <= len(pieces) <= PIECE_MAX_COUNT:
        print("碎片数量无效")

    # 全等矩形快车道:直接摆进目标槽位(布局按实测尺寸选择),跳过逐组合评分。
    fast_transforms = second_question_congruent_rect_transforms(pieces)
    if fast_transforms is not None:
        transforms = fast_transforms
        matches = ()
        cut_mode = "equal_rectangles"
        print("第二问完成: 等效矩形模式")
    else:
        candidates = second_question_candidate_matchings(pieces)
        full_count = len([match for match in candidates if tuple(match[5:]) == (0.0, 1.0, 0.0, 1.0)])
        partial_count = len(candidates) - full_count
        preferred_pairs = second_question_first_template_allowed_pairs(first_template_match)
        guided_match_sets = second_question_first_template_guided_match_sets(
            pieces, first_template_match) if first_template_match is not None else None
        ring_pairs, ring_priorities = second_question_first_template_ring_priorities(
            pieces, candidates, first_template_match) if first_template_match is not None else (None, None)
        first_template_confident = second_question_first_template_is_confident(first_template_match)
        search_passes = []
        if guided_match_sets:
            guided_pairs = {
                tuple(sorted((item[1], item[3])))
                for match_set in guided_match_sets for item in match_set
            }
            search_passes.append((
                "first_template_guided", guided_pairs, None,
                len(guided_match_sets), False,
            ))
        if ring_pairs is not None:
            search_passes.append((
                "first_template_ring",
                ring_pairs,
                SECOND_Q_FIRST_TEMPLATE_RING_CANDIDATES_PER_PAIR,
                SECOND_Q_FIRST_TEMPLATE_RING_MAX_SCORED_COMBOS,
                False,
            ))
        if preferred_pairs is not None:
            search_passes.append((
                "first_template_topology",
                preferred_pairs,
                SECOND_Q_FIRST_TEMPLATE_CANDIDATES_PER_PAIR,
                SECOND_Q_FIRST_TEMPLATE_MAX_SCORED_COMBOS,
                True,
            ))
        search_passes.append(("auto", None, None, SECOND_Q_MAX_SCORED_COMBOS, False))
        if SECOND_Q_DEBUG_CANDIDATES:
            print("第二问碎片: 版本=%s 碎片数=%d 候选=%d 完整=%d 局部=%d 模式=%s" % (
                SECOND_Q_SOLVER_DEBUG_VERSION,
                len(pieces),
                len(candidates),
                full_count,
                partial_count,
                ",".join(mode for mode, _pairs, _limit, _max_scored, _tree_only in search_passes),
            ))
            second_question_print_piece_edges(pieces)
            second_question_print_candidate_edges(pieces, candidates)
        best = None
        top_candidates = []
        tried = 0
        accepted = 0
        reject_stats = {}
        solve_start_ms = time.ticks_ms()
        solve_total_start_ms = solve_start_ms
        solve_timed_out = False
        first_template_fast_ok = False
        for cut_mode, allowed_pairs, pair_limit, max_scored, tree_only in search_passes:
            solve_timed_out = False
            mode_start_tried = tried
            mode_start_accepted = accepted
            if allowed_pairs is None:
                print("第二问模式开始: %s" % cut_mode)
            else:
                print("第二问模式开始: %s 对=%s" % (
                    cut_mode,
                    ",".join("P%d-P%d" % pair for pair in sorted(allowed_pairs)),
                ))
            if cut_mode == "first_template_guided":
                combo_iter = iter(guided_match_sets)
            else:
                combo_iter = second_question_matching_sets(
                    pieces,
                    candidates,
                    cut_mode,
                    max_scored=max_scored,
                    allowed_pairs=allowed_pairs,
                    pair_candidate_limit=pair_limit,
                    tree_only=tree_only,
                    cycle_only=cut_mode.startswith("first_template_"),
                    candidate_priorities=(
                        ring_priorities if cut_mode == "first_template_ring" else None
                    ),
                )
            enum_elapsed_ms = int(time.ticks_ms() - solve_start_ms)
            print("第二问组合: 模式=%s 最大=%d 枚举=%d毫秒" % (
                cut_mode, int(max_scored), enum_elapsed_ms))
            solve_start_ms = time.ticks_ms()
            auto_geometry_heap = []
            auto_geometry_sequence = 0
            auto_geometry_reject_stats = {}
            auto_coarse_heap = []
            auto_coarse_sequence = 0
            auto_coarse_reject_stats = {}
            for matches in combo_iter:
                tried += 1
                # 被硬规则拒绝的候选也必须计时，避免 accepted=0 时无限运行。
                if (
                    SECOND_Q_SOLVE_TIME_LIMIT_MS > 0
                    and time.ticks_ms() - solve_total_start_ms > SECOND_Q_SOLVE_TIME_LIMIT_MS
                ):
                    print("第二问总超时: 模式=%s 已试=%d 已接受=%d" % (cut_mode, tried, accepted))
                    solve_timed_out = True
                    break
                if SECOND_Q_SOLVE_PROGRESS_INTERVAL > 0 and tried % SECOND_Q_SOLVE_PROGRESS_INTERVAL == 0:
                    print(
                        "第二问求解: 模式=%s 已试=%d 已接受=%d 最佳=%s"
                        % (
                            cut_mode,
                            tried,
                            accepted,
                            "none" if best is None else "%.1f" % float(best[0]),
                        )
                    )
                    if (
                        SECOND_Q_SOLVE_TIME_LIMIT_MS > 0
                        and time.ticks_ms() - solve_total_start_ms > SECOND_Q_SOLVE_TIME_LIMIT_MS
                    ):
                        print("第二问超时: 模式=%s 已试=%d 已接受=%d" % (cut_mode, tried, accepted))
                        solve_timed_out = True
                        break
                if cut_mode == "auto" and len(pieces) == 2:
                    # 两片题候选很少，直接使用完整矩形评分。粗几何预筛主要为
                    # 四片数千组合提速，可能误杀凹凸折线接缝，且在这里没有收益。
                    result = second_question_assemble_from_matches(
                        pieces,
                        matches,
                        reject_stats,
                        optimize_cycle=False,
                        use_fixed_rect_score=False,
                    )
                elif cut_mode == "auto":
                    result = second_question_assemble_from_matches(
                        pieces,
                        matches,
                        auto_geometry_reject_stats,
                        optimize_cycle=False,
                        use_fixed_rect_score=False,
                        coarse_score=True,
                        geometry_score=True,
                    )
                else:
                    result = second_question_assemble_from_matches(
                        pieces,
                        matches,
                        reject_stats,
                        optimize_cycle=cut_mode.startswith("first_template_"),
                        use_fixed_rect_score=cut_mode.startswith("first_template_"),
                    )
                if result is None:
                    continue
                if cut_mode == "auto" and len(pieces) == 2:
                    if not second_question_candidate_clearance_ok(
                        result, matches, reject_stats, strict_gap=False):
                        continue
                    accepted += 1
                    if best is None or result[0] < best[0]:
                        best = (result[0], result[1], result[2], matches, cut_mode, result[3])
                    top_candidates.append((result[0], cut_mode, matches, result[3]))
                    top_candidates.sort(key=lambda item: item[0])
                    if len(top_candidates) > SECOND_Q_DEBUG_TOP_N:
                        top_candidates.pop()
                    continue
                if cut_mode == "auto":
                    geometry_entry = (
                        -float(result[0]),
                        auto_geometry_sequence,
                        matches,
                        result[1],
                        result[2],
                        float(result[3].get("closure", 0.0)),
                    )
                    auto_geometry_sequence += 1
                    if len(auto_geometry_heap) < SECOND_Q_AUTO_GEOMETRY_FINALISTS:
                        heapq.heappush(auto_geometry_heap, geometry_entry)
                    elif geometry_entry > auto_geometry_heap[0]:
                        heapq.heapreplace(auto_geometry_heap, geometry_entry)
                    continue
                if not second_question_candidate_clearance_ok(
                    result,
                    matches,
                    reject_stats,
                    strict_gap=cut_mode.startswith("first_template_"),
                ):
                    continue
                accepted += 1
                if best is None or result[0] < best[0]:
                    best = (result[0], result[1], result[2], matches, cut_mode, result[3])
                top_candidates.append((result[0], cut_mode, matches, result[3]))
                top_candidates.sort(key=lambda item: item[0])
                if len(top_candidates) > SECOND_Q_DEBUG_TOP_N:
                    top_candidates.pop()
                if cut_mode.startswith("first_template_") and first_template_confident:
                    detail = result[3]
                    piece_area = max(1.0, sum(
                        abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
                        for polygon in result[2]
                    ))
                    geometric_overlap = second_question_pairwise_overlap_area(result[2])
                    overlap_ratio = geometric_overlap / piece_area
                    if (
                        float(detail.get("fill", 0.0)) >= SECOND_Q_FIRST_TEMPLATE_EARLY_FILL
                        and overlap_ratio <= SECOND_Q_FIRST_TEMPLATE_EARLY_OVERLAP_RATIO
                        and float(detail.get("boundary_cover", 1.0))
                        <= SECOND_Q_FIRST_TEMPLATE_EARLY_MAX_BOUNDARY_MISSING
                    ):
                        first_template_fast_ok = True
                        print(
                            "SECOND Q FIRST TEMPLATE FAST OK fill=%.3f overlap=%.3f boundary=%.3f tried=%d"
                            % (
                                float(detail.get("fill", 0.0)),
                                overlap_ratio,
                                float(detail.get("boundary_cover", 1.0)),
                                tried,
                            )
                        )
                        break
                if (
                    SECOND_Q_SOLVE_TIME_LIMIT_MS > 0
                    and time.ticks_ms() - solve_total_start_ms > SECOND_Q_SOLVE_TIME_LIMIT_MS
                ):
                    print("第二问超时: 模式=%s 已试=%d 已接受=%d" % (cut_mode, tried, accepted))
                    solve_timed_out = True
                    break
            if cut_mode == "auto" and auto_geometry_heap:
                geometry_elapsed_ms = int(time.ticks_ms() - solve_start_ms)
                geometry_finalists = sorted(
                    auto_geometry_heap, key=lambda item: (-item[0], item[1]))
                print(
                    "第二问几何预筛结束: 保留=%d 拒绝=%s 耗时=%d毫秒"
                    % (
                        len(geometry_finalists),
                        second_question_format_reject_stats(auto_geometry_reject_stats),
                        geometry_elapsed_ms,
                    )
                )
                coarse_start_ms = time.ticks_ms()
                for _neg_score, _sequence, matches, transforms, assembled, closure_error in geometry_finalists:
                    result = second_question_score_assembly(
                        transforms,
                        assembled,
                        matches,
                        closure_error,
                        auto_coarse_reject_stats,
                        apply_rectlike_reject=False,
                        use_fixed_rect_score=False,
                        mask_scale=SECOND_Q_AUTO_COARSE_MASK_SCALE,
                        coarse_score=True,
                    )
                    if result is None:
                        continue
                    coarse_entry = (
                        -float(result[0]),
                        auto_coarse_sequence,
                        matches,
                        result[1],
                        result[2],
                        float(result[3].get("closure", 0.0)),
                    )
                    auto_coarse_sequence += 1
                    if len(auto_coarse_heap) < SECOND_Q_AUTO_FINALISTS:
                        heapq.heappush(auto_coarse_heap, coarse_entry)
                    elif coarse_entry > auto_coarse_heap[0]:
                        heapq.heapreplace(auto_coarse_heap, coarse_entry)

            if cut_mode == "auto" and len(pieces) > 2 and not auto_geometry_heap:
                for reason, count in auto_geometry_reject_stats.items():
                    reject_stats[reason] = reject_stats.get(reason, 0) + count
                print(
                    "第二问几何预筛无候选: 拒绝=%s"
                    % second_question_format_reject_stats(auto_geometry_reject_stats)
                )

            if cut_mode == "auto" and auto_coarse_heap:
                coarse_elapsed_ms = int(time.ticks_ms() - coarse_start_ms)
                finalists = sorted(auto_coarse_heap, key=lambda item: (-item[0], item[1]))
                print(
                    "第二问粗筛结束: 保留=%d 拒绝=%s 耗时=%d毫秒"
                    % (
                        len(finalists),
                        second_question_format_reject_stats(auto_coarse_reject_stats),
                        coarse_elapsed_ms,
                    )
                )
                final_start_ms = time.ticks_ms()
                for _neg_score, _sequence, matches, transforms, assembled, closure_error in finalists:
                    result = second_question_score_assembly(
                        transforms,
                        assembled,
                        matches,
                        closure_error,
                        reject_stats,
                        use_fixed_rect_score=False,
                    )
                    if result is None:
                        continue
                    if not second_question_candidate_clearance_ok(
                        result, matches, reject_stats, strict_gap=False):
                        continue
                    accepted += 1
                    if best is None or result[0] < best[0]:
                        best = (result[0], result[1], result[2], matches, cut_mode, result[3])
                    top_candidates.append((result[0], cut_mode, matches, result[3]))
                    top_candidates.sort(key=lambda item: item[0])
                    if len(top_candidates) > SECOND_Q_DEBUG_TOP_N:
                        top_candidates.pop()
                print(
                    "第二问精评结束: 候选=%d 接受=%d 耗时=%d毫秒"
                    % (
                        len(finalists),
                        accepted - mode_start_accepted,
                        int(time.ticks_ms() - final_start_ms),
                    )
                )
            elapsed_ms = time.ticks_ms() - solve_start_ms
            print(
                "第二问模式结束: %s 已试=%d 已接受=%d 耗时=%d毫秒"
                % (cut_mode, tried - mode_start_tried, accepted - mode_start_accepted, int(elapsed_ms))
            )
            if cut_mode.startswith("first_template_"):
                if first_template_fast_ok:
                    print("第一问模板高置信拓扑成功; 跳过自动回退")
                    break
                print("第一问模板快路未达到高质量门槛; 继续通用回退")
            if solve_timed_out:
                break
        if best is None:
            raise RuntimeError("第二问求解失败: 模式=%s 候选=%d 完整=%d 局部=%d 已试=%d 已接受=%d 拒绝原因=%s" % (
                SECOND_Q_CUT_MODE,
                len(candidates),
                full_count,
                partial_count,
                tried,
                accepted,
                second_question_format_reject_stats(reject_stats),
            ))

        second_question_print_top_candidates(top_candidates)

        _score, transforms, _assembled, matches, cut_mode, _detail = best
        print("第二问完成: 模式=%s 分数=%.1f 匹配=%d 已试=%d 已接受=%d" % (
            cut_mode,
            float(_score),
            len(matches),
            tried,
            accepted,
        ))
    assembled = [apply_homography_points(piece, transform) for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(assembled).astype(np.float32)
    rect = cv2.minAreaRect(all_points)
    angle = float(rect[2])
    size_w, size_h = rect[1]
    if size_w < size_h:
        angle += 90.0
    normalize = rigid_transform(math.radians(-angle), 0.0, 0.0)
    rotated = apply_homography_points(all_points, normalize)
    min_point = rotated.min(axis=0)
    max_point = rotated.max(axis=0)
    if (max_point - min_point)[0] < (max_point - min_point)[1]:
        normalize = rigid_transform(math.radians(90.0 - angle), 0.0, 0.0)
        rotated = apply_homography_points(all_points, normalize)
        min_point = rotated.min(axis=0)
        max_point = rotated.max(axis=0)
    if (max_point - min_point)[0] > (max_point - min_point)[1]:
        rect_center = (min_point + max_point) * 0.5
        rotate_vertical = rigid_transform(math.radians(90.0), 0.0, 0.0)
        rotated_center = apply_homography_points(np.asarray([rect_center], dtype=np.float32), rotate_vertical)[0]
        keep_center = rigid_transform(0.0, float(rect_center[0] - rotated_center[0]), float(rect_center[1] - rotated_center[1]))
        normalize = keep_center.dot(rotate_vertical).dot(normalize)
        rotated = apply_homography_points(all_points, normalize)
        min_point = rotated.min(axis=0)
        max_point = rotated.max(axis=0)
    target_size = np.maximum(1.0, max_point - min_point)
    target_rect = order_points(cv2.boxPoints(cv2.minAreaRect(rotated.astype(np.float32))))
    return transforms, normalize, min_point, target_size, target_rect, matches


def second_question_geometric_contact_normals(polygon_i, polygon_j, center_delta):
    """从最终几何轮廓中找出两片之间全部近共线接缝法线。"""
    polygon_i = np.asarray(polygon_i, dtype=np.float32).reshape(-1, 2)
    polygon_j = np.asarray(polygon_j, dtype=np.float32).reshape(-1, 2)
    candidates = []
    for edge_i_start, edge_i_end in polygon_edges(polygon_i):
        tangent_i = np.asarray(edge_i_end - edge_i_start, dtype=np.float32)
        length_i = float(np.linalg.norm(tangent_i))
        if length_i <= 1e-6:
            continue
        tangent_i /= length_i
        normal_i = np.asarray([-tangent_i[1], tangent_i[0]], dtype=np.float32)
        midpoint_i = 0.5 * (edge_i_start + edge_i_end)
        projection_i = sorted((
            float(np.dot(edge_i_start, tangent_i)),
            float(np.dot(edge_i_end, tangent_i)),
        ))
        for edge_j_start, edge_j_end in polygon_edges(polygon_j):
            tangent_j = np.asarray(edge_j_end - edge_j_start, dtype=np.float32)
            length_j = float(np.linalg.norm(tangent_j))
            if length_j <= 1e-6:
                continue
            tangent_j /= length_j
            parallel = abs(float(np.dot(tangent_i, tangent_j)))
            if parallel < SECOND_Q_CLEARANCE_CONTACT_PARALLEL_MIN:
                continue
            projection_j = sorted((
                float(np.dot(edge_j_start, tangent_i)),
                float(np.dot(edge_j_end, tangent_i)),
            ))
            overlap = min(projection_i[1], projection_j[1]) - max(
                projection_i[0], projection_j[0])
            if overlap < SECOND_Q_CLEARANCE_CONTACT_OVERLAP_PX:
                continue
            midpoint_j = 0.5 * (edge_j_start + edge_j_end)
            line_distance = abs(float(np.dot(midpoint_j - midpoint_i, normal_i)))
            if line_distance > SECOND_Q_CLEARANCE_CONTACT_DISTANCE_PX:
                continue
            normal = normal_i.copy()
            if float(np.dot(normal, center_delta)) < 0.0:
                normal = -normal
            candidates.append((line_distance, -overlap, -parallel, normal))

    candidates.sort(key=lambda item: item[:3])
    normals = []
    for _distance, _overlap, _parallel, normal in candidates:
        if any(float(np.dot(normal, previous)) >= 0.985 for previous in normals):
            continue
        normals.append(normal)
        if len(normals) >= 3:
            break
    return normals


def second_question_clearance_pair_normals(assembled, matches):
    """返回全部实际接缝约束；同一碎片对可保留多个折线法线。"""
    polygons = [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in assembled]
    centers = [np.mean(polygon, axis=0) for polygon in polygons]
    pair_normals = {}

    def add_normal(key, normal):
        normal = np.asarray(normal, dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-6:
            return
        normal /= normal_norm
        center_delta = centers[key[1]] - centers[key[0]]
        if float(np.dot(normal, center_delta)) < 0.0:
            normal = -normal
        normals = pair_normals.setdefault(key, [])
        if any(float(np.dot(normal, previous)) >= 0.985 for previous in normals):
            return
        normals.append(normal)

    for match in matches or ():
        _, piece_i, edge_i, piece_j, _edge_j = match[:5]
        key = tuple(sorted((int(piece_i), int(piece_j))))
        edge_start, edge_end = second_question_resolve_edge(polygons[piece_i], edge_i)
        tangent = np.asarray(edge_end - edge_start, dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-6:
            continue
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / tangent_norm
        add_normal(key, normal)

    # 树形求解只声明 N-1 条边，但最终矩形里可能存在额外闭环接触；
    # 同一对凹凸碎片也可能共享两段非共线折线，必须逐段加入约束。
    for piece_i, piece_j in itertools.combinations(range(len(polygons)), 2):
        key = (piece_i, piece_j)
        center_delta = centers[piece_j] - centers[piece_i]
        for normal in second_question_geometric_contact_normals(
            polygons[piece_i], polygons[piece_j], center_delta):
            add_normal(key, normal)

    if pair_normals:
        return [
            (key[0], key[1], normal)
            for key, normals in sorted(pair_normals.items())
            for normal in normals
        ]

    # 等分矩形快车道没有显式 matches；用中心距离最小生成树分离所有行列。
    edges = []
    for piece_i, piece_j in itertools.combinations(range(len(polygons)), 2):
        delta = centers[piece_j] - centers[piece_i]
        length = float(np.linalg.norm(delta))
        if length > 1e-6:
            edges.append((length, piece_i, piece_j, delta / length))
    edges.sort(key=lambda item: item[0])
    parent = list(range(len(polygons)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    result = []
    for _length, piece_i, piece_j, normal in edges:
        root_i = find(piece_i)
        root_j = find(piece_j)
        if root_i == root_j:
            continue
        parent[root_j] = root_i
        result.append((piece_i, piece_j, normal))
        if len(result) >= len(polygons) - 1:
            break
    return result


def second_question_clearance_offsets(assembled, matches, gap_px):
    """迭代投影求平移量，使每条已匹配接缝沿法线至少分开固定距离。"""
    count = len(assembled)
    pairs = second_question_clearance_pair_normals(assembled, matches)
    if count <= 1 or not pairs or gap_px <= 0.0:
        return np.zeros((count, 2), dtype=np.float32)

    offsets = np.zeros((count, 2), dtype=np.float32)
    tolerance = 0.01
    for _iteration in range(40):
        max_deficit = 0.0
        for piece_i, piece_j, normal in pairs:
            projected_gap = float(np.dot(offsets[piece_j] - offsets[piece_i], normal))
            deficit = float(gap_px) - projected_gap
            if deficit <= tolerance:
                continue
            correction = np.asarray(normal, dtype=np.float32) * (0.5 * deficit)
            offsets[piece_i] -= correction
            offsets[piece_j] += correction
            max_deficit = max(max_deficit, deficit)
        offsets -= np.mean(offsets, axis=0, keepdims=True)
        if max_deficit <= tolerance:
            break
    offsets -= np.mean(offsets, axis=0, keepdims=True)
    return offsets


def second_question_clearance_achieved_gap(assembled, matches, offsets):
    pairs = second_question_clearance_pair_normals(assembled, matches)
    if not pairs:
        return 0.0
    return max(0.0, min(
        float(np.dot(offsets[piece_j] - offsets[piece_i], normal))
        for piece_i, piece_j, normal in pairs
    ))


def second_question_apply_clearance(assembled, matches, requested_gap_px):
    """应用固定间距；若拓扑约束冲突导致重叠，逐步缩小间距直至安全。"""
    polygons = [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in assembled]
    gap_px = max(0.0, float(requested_gap_px))
    for _attempt in range(5):
        offsets = second_question_clearance_offsets(polygons, matches, gap_px)
        shifted = [polygon + offsets[index] for index, polygon in enumerate(polygons)]
        overlap = second_question_pairwise_overlap_area(shifted)
        if overlap <= 1.0:
            achieved_gap = second_question_clearance_achieved_gap(polygons, matches, offsets)
            return offsets, shifted, min(gap_px, achieved_gap), overlap
        gap_px *= 0.5
    offsets = np.zeros((len(polygons), 2), dtype=np.float32)
    return offsets, polygons, 0.0, second_question_pairwise_overlap_area(polygons)


def attach_second_question_targets(pieces):
    if not SECOND_QUESTION_MODE or len(pieces) < 2:
        return False

    for piece in pieces:
        normalized_polygon = second_question_normalize_polygon_winding(piece.get("polygon", []))
        piece["polygon"] = normalized_polygon.tolist()
    first_template_match = second_question_first_template_match(pieces)
    if first_template_match is not None:
        summary = []
        for piece_index, template_index in enumerate(first_template_match["assignment"]):
            summary.append("P%d=%s" % (piece_index, FIRST_Q_TEMPLATES[template_index]["name"]))
        print(
            "第二问检测到第一问模板: 成本=%.3f 最大形状分=%.3f 碎片分配=%s"
            % (
                float(first_template_match["cost"]),
                float(first_template_match["max_shape_score"]),
                " ".join(summary),
            )
        )

    target_side = choose_second_question_target_side(pieces)
    polygons = [piece.get("polygon", []) for piece in pieces]
    try:
        transforms, normalize, min_point, target_size, target_rect_points, _matches = second_question_solve_transforms(
            polygons,
            first_template_match=first_template_match,
        )
    except Exception as err:
        print("第二问求解失败: %s" % err)
        return False

    target_size = np.asarray(target_size, dtype=np.float32)
    min_point = np.asarray(min_point, dtype=np.float32)

    # 求解和矩形验收保持真实尺寸；这里只为机械放置增加固定毫米安全间距。
    normalized_transforms = [normalize.dot(transform) for transform in transforms]
    normalized_polygons = [
        apply_homography_points(np.asarray(piece.get("polygon", []), dtype=np.float32), transform)
        for piece, transform in zip(pieces, normalized_transforms)
    ]
    px_per_mm = 0.5 * (
        (WARP_W - 1.0) / A4_W_MM
        + (WARP_H - 1.0) / A4_H_MM
    )
    requested_gap_px = float(SECOND_Q_TARGET_GAP_MM) * px_per_mm
    clearance_constraint_count = len(
        second_question_clearance_pair_normals(normalized_polygons, _matches))
    clearance_offsets, clearance_polygons, actual_gap_px, clearance_overlap = second_question_apply_clearance(
        normalized_polygons,
        _matches,
        requested_gap_px,
    )
    clearance_points = np.vstack(clearance_polygons).astype(np.float32)
    clearance_min = clearance_points.min(axis=0)
    clearance_max = clearance_points.max(axis=0)
    clearance_size = np.maximum(1.0, clearance_max - clearance_min)
    available_size = second_question_target_available_size(target_side)
    if np.any(clearance_size > available_size + 1e-3):
        print(
            "SECOND Q GAP DISABLED no_space size=%.0fx%.0f avail=%.0fx%.0f"
            % (
                float(clearance_size[0]),
                float(clearance_size[1]),
                float(available_size[0]),
                float(available_size[1]),
            )
        )
        clearance_offsets = np.zeros((len(pieces), 2), dtype=np.float32)
        clearance_polygons = normalized_polygons
        actual_gap_px = 0.0
        clearance_overlap = second_question_pairwise_overlap_area(clearance_polygons)
        clearance_points = np.vstack(clearance_polygons).astype(np.float32)
        clearance_min = clearance_points.min(axis=0)
        clearance_max = clearance_points.max(axis=0)
        clearance_size = np.maximum(1.0, clearance_max - clearance_min)

    target_origin = second_question_target_origin(target_side, clearance_size)
    translate = rigid_transform(
        0.0,
        float(target_origin[0] - clearance_min[0]),
        float(target_origin[1] - clearance_min[1]),
    )
    target_rect_points = np.asarray(target_rect_points, dtype=np.float32).reshape(-1, 2)
    target_rect = np.rint(
        target_rect_points + (target_origin - clearance_min).reshape(1, 2)
    ).astype(np.int32).tolist()
    print(
        "SECOND Q GAP requested=%.1fmm actual=%.1fmm overlap=%.1f constraints=%d"
        % (
            float(SECOND_Q_TARGET_GAP_MM),
            float(actual_gap_px / max(1e-6, px_per_mm)),
            float(clearance_overlap),
            int(clearance_constraint_count),
        )
    )

    for piece_index, piece in enumerate(pieces):
        offset_transform = rigid_transform(
            0.0,
            float(clearance_offsets[piece_index][0]),
            float(clearance_offsets[piece_index][1]),
        )
        transform = translate.dot(offset_transform).dot(normalized_transforms[piece_index])
        polygon = np.asarray(piece.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        target_polygon = apply_homography_points(polygon, transform)
        current_center = np.asarray(piece.get("center", polygon_center(polygon)), dtype=np.float32).reshape(1, 2)
        target_center_point = apply_homography_points(current_center, transform)[0]
        rotate_deg = normalize_angle_180(math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0]))))
        piece_name = "P%d" % piece_index
        target_polygon_list = np.rint(target_polygon).astype(np.int32).tolist()
        target_center_list = [int(round(float(target_center_point[0]))), int(round(float(target_center_point[1])))]
        piece["id"] = piece_index
        piece["template"] = piece_name
        piece["target_side"] = target_side
        piece["target_size"] = [float(clearance_size[0]), float(clearance_size[1])]
        piece["target_polygon"] = target_rect
        piece["target_center"] = target_center_list
        piece["target_detected_polygon"] = target_polygon_list
        piece["final_polygon"] = target_rect
        piece["final_center"] = target_center_list
        piece["final_detected_polygon"] = target_polygon_list
        piece["rotate_deg"] = round(rotate_deg, 1)
        piece["rotate_method"] = "q2_gap_mm"
        piece["move"] = {
            "pick": piece.get("center", [0, 0]),
            "place": target_center_list,
            "final_place": target_center_list,
            "rotate_deg": round(rotate_deg, 1),
            "rotate_method": "q2_gap_mm",
        }
    return True


# =========================
# Move Planning and Capture Aggregation
# =========================
# 把第一/第二问求解结果转换为机械端需要的抓取点、放置点和旋转角。
def mech_mm_offset_to_warp_pixels(offset_x_mm, offset_y_mm):
    """把沿 A4 标准透视图方向的毫米补偿转换为像素。"""
    return np.asarray([
        float(offset_x_mm) * (WARP_W - 1.0) / A4_W_MM,
        float(offset_y_mm) * (WARP_H - 1.0) / A4_H_MM,
    ], dtype=np.float32)


def compensate_global_mech_mapping(point):
    """统一修正视觉坐标到 STM32 坐标的比例、旋转和原点偏差。"""
    point = np.asarray(point, dtype=np.float32)
    center = np.asarray([
        (WARP_W - 1.0) * 0.5,
        (WARP_H - 1.0) * 0.5,
    ], dtype=np.float32)
    relative = (point - center) * np.asarray([
        float(MECH_GLOBAL_X_SCALE),
        float(MECH_GLOBAL_Y_SCALE),
    ], dtype=np.float32)
    angle_rad = math.radians(float(MECH_GLOBAL_ROTATION_DEG))
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    corrected = center + np.asarray([
        cosine * float(relative[0]) - sine * float(relative[1]),
        sine * float(relative[0]) + cosine * float(relative[1]),
    ], dtype=np.float32)
    corrected += mech_mm_offset_to_warp_pixels(
        MECH_GLOBAL_FIXED_OFFSET_X_MM,
        MECH_GLOBAL_FIXED_OFFSET_Y_MM,
    )
    corrected[0] = np.clip(corrected[0], 0.0, WARP_W - 1.0)
    corrected[1] = np.clip(corrected[1], 0.0, WARP_H - 1.0)
    return corrected


def build_move_plan(pieces):
    pending = []
    for piece in pieces:
        move = piece.get("move")
        if not move:
            continue
        pick = move["pick"]
        place = move.get("final_place", move["place"])
        pending.append((piece, move, pick, place))

    pick_fixed_offset = mech_mm_offset_to_warp_pixels(
        MECH_PICK_FIXED_OFFSET_X_MM,
        MECH_PICK_FIXED_OFFSET_Y_MM,
    )
    place_fixed_offset = mech_mm_offset_to_warp_pixels(
        MECH_PLACE_FIXED_OFFSET_X_MM,
        MECH_PLACE_FIXED_OFFSET_Y_MM,
    )

    plan = []
    for piece, move, pick, place in pending:
        pick_for_output = compensate_global_mech_mapping(pick) + pick_fixed_offset
        pick_for_output[0] = np.clip(pick_for_output[0], 0.0, WARP_W - 1.0)
        pick_for_output[1] = np.clip(pick_for_output[1], 0.0, WARP_H - 1.0)

        place_for_output = compensate_global_mech_mapping(place) + place_fixed_offset

        plan.append({
            "piece": piece.get("template", "?"),
            "pick": pick,
            "place": move["place"],
            "final_place": place,
            "pick_mech": warp_to_mech_point(pick_for_output),
            "place_mech": warp_to_mech_point(place_for_output),
            "pick_output_warp": np.asarray(pick_for_output, dtype=np.float32).tolist(),
            "place_output_warp": np.asarray(place_for_output, dtype=np.float32).tolist(),
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
    if SECOND_QUESTION_MODE:
        return aggregate_second_question_capture_samples(valid_samples)

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
        "a4_method": capture_a4_method(valid_samples),
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


def aggregate_second_question_capture_samples(valid_samples):
    count_histogram = {}
    for sample in valid_samples:
        piece_count = len(sample.get("pieces", []))
        if 2 <= piece_count <= PIECE_MAX_COUNT:
            count_histogram[piece_count] = count_histogram.get(piece_count, 0) + 1
    if not count_histogram:
        observed = {}
        for sample in valid_samples:
            piece_count = len(sample.get("pieces", []))
            observed[piece_count] = observed.get(piece_count, 0) + 1
        print("第二问失败: 碎片数量分布=%s 需要=2..%d" % (observed, PIECE_MAX_COUNT))
        return None, []
    expected_count = max(count_histogram.items(), key=lambda item: item[1])[0]
    samples = [sample for sample in valid_samples if len(sample.get("pieces", [])) == expected_count]
    print("第二问捕获分布=%s 采用=%d 帧=%d" % (
        count_histogram, expected_count, len(samples)))
    if len(samples) < CAPTURE_MIN_VALID_FRAMES:
        print("第二问失败: 稳定帧数=%d 总帧数=%d" % (expected_count, len(samples)))
        return None, []

    base_sample = choose_capture_medoid_sample(samples)
    base_pieces = sorted(
        base_sample.get("pieces", []),
        key=lambda piece: (piece["center"][1], piece["center"][0]),
    )
    grouped = [[base_piece] for base_piece in base_pieces]
    base_centers = [np.asarray(piece["center"], dtype=np.float32) for piece in base_pieces]
    matched_samples = 1
    for sample in samples:
        if sample is base_sample:
            continue
        pieces = sorted(sample.get("pieces", []), key=lambda piece: (piece["center"][1], piece["center"][0]))
        order = best_piece_order_by_center(base_centers, pieces)
        if order is None:
            continue
        matched_samples += 1
        for group_index, piece_index in enumerate(order):
            grouped[group_index].append(pieces[piece_index])
    print("第二问聚合配对: 匹配帧=%d/%d 分组=%s" % (
        matched_samples, len(samples), [len(group) for group in grouped]))

    aggregated_pieces = []
    for piece_index, group in enumerate(grouped):
        if len(group) < CAPTURE_MIN_VALID_FRAMES:
            continue
        filtered = filter_second_question_piece_samples(group)
        if len(filtered) < CAPTURE_MIN_VALID_FRAMES:
            continue
        aggregated_pieces.append(average_second_question_piece_samples(piece_index, filtered))

    if len(aggregated_pieces) != expected_count:
        print("第二问失败: 聚合碎片数=%d 期望=%d" % (len(aggregated_pieces), expected_count))
        return None, []
    if not attach_second_question_targets(aggregated_pieces):
        print("第二问失败: 求解碎片数=%d" % len(aggregated_pieces))
        return None, []

    corners = average_a4_corners(samples)
    result = {
        "status": True,
        "stable": True,
        "a4_method": capture_a4_method(samples),
        "question_mode": QUESTION_MODE,
        "corners": corners,
        "midline": standard_midline(),
        "pieces_count": len(aggregated_pieces),
        "pieces": aggregated_pieces,
        "move_plan": build_move_plan(aggregated_pieces),
        "capture_frames": len(samples),
    }
    piece_contours = [
        np.asarray(piece["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        for piece in aggregated_pieces
    ]
    return result, piece_contours


def capture_a4_method(samples):
    counts = {}
    for sample in samples:
        method = sample.get("a4_method", "unknown")
        counts[method] = counts.get(method, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def best_piece_order_by_center(base_centers, pieces):
    if len(base_centers) != len(pieces):
        return None
    base = np.asarray(base_centers, dtype=np.float32).reshape(-1, 2)
    centers = np.asarray(
        [piece["center"] for piece in pieces], dtype=np.float32).reshape(-1, 2)
    base_centered = base - np.mean(base, axis=0, keepdims=True)
    centers_centered = centers - np.mean(centers, axis=0, keepdims=True)
    best_order = None
    best_error = None
    for order in itertools.permutations(range(len(pieces))):
        ordered = centers_centered[np.asarray(order, dtype=np.int32)]
        error = float(np.sum(np.linalg.norm(base_centered - ordered, axis=1)))
        if best_error is None or error < best_error:
            best_error = error
            best_order = order
    if best_error is not None and best_error > CAPTURE_OUTLIER_CENTER_PX * len(pieces):
        return None
    return best_order


def choose_capture_medoid_sample(samples):
    """选择最接近全部有效帧中位布局的基准帧，避免首帧异常拖垮聚合。"""
    if len(samples) <= 1:
        return samples[0]
    signatures = []
    for sample in samples:
        centers = np.asarray(
            [piece.get("center", [0, 0]) for piece in sample.get("pieces", [])],
            dtype=np.float32,
        ).reshape(-1, 2)
        order = np.lexsort((centers[:, 0], centers[:, 1]))
        centers = centers[order]
        signatures.append(centers - np.mean(centers, axis=0, keepdims=True))
    signature_array = np.asarray(signatures, dtype=np.float32)
    median_signature = np.median(signature_array, axis=0)
    errors = np.sum(np.linalg.norm(signature_array - median_signature, axis=2), axis=1)
    return samples[int(np.argmin(errors))]


def filter_second_question_piece_samples(pieces):
    centers = np.asarray([piece.get("center", [0, 0]) for piece in pieces], dtype=np.float32)
    median_center = np.median(centers, axis=0)
    filtered = []
    for piece, center in zip(pieces, centers):
        center_error = float(np.linalg.norm(center - median_center))
        if center_error <= CAPTURE_OUTLIER_CENTER_PX:
            filtered.append(piece)
    return filtered


def average_second_question_piece_samples(piece_index, pieces):
    center = average_points([piece.get("center", [0, 0]) for piece in pieces])
    polygon = average_polygon_field(pieces, "polygon") or pieces[0].get("polygon", [])
    expanded_polygon = average_polygon_field(pieces, "expanded_polygon") or polygon
    return {
        "id": piece_index,
        "area": int(round(float(np.mean([piece.get("area", 0) for piece in pieces])))),
        "source_area": int(round(float(np.mean([piece.get("source_area", 0) for piece in pieces])))),
        "center": center,
        "bbox": bbox_from_polygon(polygon),
        "side": "left" if center[0] < WARP_W // 2 else "right",
        "points": len(polygon),
        "polygon": polygon,
        "expanded_polygon": expanded_polygon,
        "template": "P%d" % piece_index,
    }


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

    # 角点拟合偶尔会多/少一个点；使用出现次数最多的边数，避免单帧异常主导结果。
    count_histogram = {}
    for polygon in polygons:
        count = len(polygon)
        count_histogram[count] = count_histogram.get(count, 0) + 1
    point_count = max(count_histogram.items(), key=lambda item: item[1])[0]
    candidates = [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in polygons if len(polygon) == point_count
    ]
    if not candidates:
        return polygons[0]

    # OpenCV 轮廓起点可能逐帧循环移动，绕向也可能翻转。先使用捕获中位帧作参考，
    # 再把每帧顶点对齐到同一索引，最后逐坐标取中位数抑制抖动和离群角点。
    def align_to_reference(reference, candidate):
        best = None
        for variant in (candidate, candidate[::-1]):
            for shift in range(point_count):
                aligned = np.roll(variant, -shift, axis=0)
                error = float(np.mean(np.sum((aligned - reference) ** 2, axis=1)))
                if best is None or error < best[0]:
                    best = (error, aligned)
        return best

    # grouped[0] 来自布局中位帧，直接作为参考可避免在 MaixCAM 上做 O(n^2) 两两比较。
    reference = candidates[0]
    aligned_candidates = [
        align_to_reference(reference, candidate)[1]
        for candidate in candidates
    ]
    median = np.median(np.asarray(aligned_candidates, dtype=np.float32), axis=0)
    return np.rint(median).astype(np.int32).tolist()


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


# =========================
# Mechanical Coordinate Output
# =========================
# 将 A4 透视图像素坐标映射到机械坐标，并按 STM32 协议格式化输出。
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


def mech_output_calibration_matrix():
    """A4 标准像素直接映射到 STM32 最终输出坐标，避免轴交换标定歧义。"""
    src = np.float32(
        [
            [0.0, 0.0],
            [WARP_W - 1.0, 0.0],
            [WARP_W - 1.0, WARP_H - 1.0],
            [0.0, WARP_H - 1.0],
        ]
    )
    return cv2.getPerspectiveTransform(
        src, np.float32(MECH_OUTPUT_CALIBRATION_POINTS))


def warp_to_mech_point(point):
    corrected = np.asarray(point, dtype=np.float32)
    if not MECH_COORD_OUTPUT_ENABLED:
        return format_output_mech_point([float(corrected[0]), float(corrected[1])])

    src = np.float32([[[float(corrected[0]), float(corrected[1])]]])
    if MECH_DIRECT_OUTPUT_CALIBRATION_ENABLED:
        dst = cv2.perspectiveTransform(src, mech_output_calibration_matrix())[0][0]
        return [float(dst[0]), float(dst[1])]

    matrix = mech_calibration_matrix()
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
    if isinstance(name, str) and name.startswith("P"):
        try:
            return int(name[1:])
        except Exception:
            return 9
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
                "pick_warp": move.get("pick", [0, 0]),
                "place_warp": move.get("final_place", move.get("place", [0, 0])),
                "pick_output_warp": move.get("pick_output_warp", move.get("pick", [0, 0])),
                "place_output_warp": move.get(
                    "place_output_warp", move.get("final_place", move.get("place", [0, 0]))),
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
        pick_warp = record["pick_warp"]
        pick_output_warp = record["pick_output_warp"]
        place_warp = record["place_warp"]
        place_output_warp = record["place_output_warp"]
        parts.append(
            "%s pick_px=(%.0f,%.0f)->(%.0f,%.0f) place_px=(%.0f,%.0f)->(%.0f,%.0f) mech=(%d,%d)->(%d,%d) rot=%.1f/%s"
            % (
                record["name"],
                float(pick_warp[0]),
                float(pick_warp[1]),
                float(pick_output_warp[0]),
                float(pick_output_warp[1]),
                float(place_warp[0]),
                float(place_warp[1]),
                float(place_output_warp[0]),
                float(place_output_warp[1]),
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
        print("串口写入失败: %s" % err)


def send_binary(serial_obj, packet, log_line=None):
    if log_line:
        print(log_line)
    else:
        print("发送 %d 字节" % len(packet))
    if serial_obj is None:
        return

    try:
        serial_obj.write(packet)
    except Exception as err:
        print("串口写入失败: %s" % err)


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
    inner_scale = float(PIECE_A4_MASK_INNER_SCALE)
    inner = center + (corners_array - center) * inner_scale
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

    if PIECE_REFINE_CORNERS_BY_LINES and best is not None and 3 <= len(best) <= PIECE_MAX_POINTS and cv2.isContourConvex(best):
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

# =========================
# Display and Debug Views
# =========================
# 现场显示和调试画面绘制：原图、透视图、碎片编号、目标位置、mask 和性能信息。
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
        color = piece_display_color(piece)
        cv2.drawContours(frame, [contour], -1, color, 2)
        cx, cy = piece["center"]
        cv2.circle(frame, (cx, cy), 4, color, -1)
        label = str(piece["id"])
        if piece.get("template"):
            label += ":" + piece["template"]
        cv2.putText(frame, label, (cx + 5, max(12, cy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def piece_preview_transform(source_polygon, target_polygon):
    source = np.asarray(source_polygon, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_polygon, dtype=np.float32).reshape(-1, 2)
    if len(source) < 3 or len(target) < 3 or len(source) != len(target):
        return None, None
    if len(source) == 3:
        matrix = cv2.getAffineTransform(source[:3], target[:3])
        return matrix, "affine"
    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None:
        return None, None
    return matrix.astype(np.float32), "homography"


def draw_target_texture_preview(frame, source_view, pieces):
    if not PREVIEW_TEXTURE_ENABLED or source_view is None:
        return
    height, width = frame.shape[:2]
    for piece in pieces or []:
        source_polygon = piece.get("polygon")
        target_polygon = (
            piece.get("target_detected_polygon")
            or piece.get("final_detected_polygon")
        )
        if not source_polygon or not target_polygon:
            continue
        transform, transform_kind = piece_preview_transform(source_polygon, target_polygon)
        if transform is None:
            continue

        source_mask = np.zeros(source_view.shape[:2], dtype=np.uint8)
        source_contour = np.asarray(source_polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(source_mask, [source_contour], 255)
        source_pixels = cv2.bitwise_and(source_view, source_view, mask=source_mask)

        if transform_kind == "affine":
            warped_pixels = cv2.warpAffine(source_pixels, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_mask = cv2.warpAffine(source_mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        else:
            warped_pixels = cv2.warpPerspective(source_pixels, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_mask = cv2.warpPerspective(source_mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        target_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        target_contour = np.asarray(target_polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(target_mask, [target_contour], 255)
        warped_mask = cv2.bitwise_and(warped_mask, target_mask)
        if PREVIEW_TEXTURE_ALPHA >= 0.999:
            frame[warped_mask > 0] = warped_pixels[warped_mask > 0]
        else:
            alpha = float(PREVIEW_TEXTURE_ALPHA)
            mask = warped_mask > 0
            frame[mask] = (frame[mask].astype(np.float32) * (1.0 - alpha) + warped_pixels[mask].astype(np.float32) * alpha).astype(np.uint8)


def draw_question_targets(frame, pieces):
    if SECOND_QUESTION_MODE:
        draw_second_question_targets(frame, pieces)
    else:
        draw_first_question_targets(frame, pieces)


def draw_second_question_targets(frame, pieces):
    if not SECOND_QUESTION_MODE or not pieces:
        return

    target_side = None
    rect_points = None
    for piece in pieces or []:
        if piece.get("target_side"):
            target_side = piece["target_side"]
        if piece.get("target_polygon"):
            rect_points = piece["target_polygon"]
        if target_side is not None and rect_points is not None:
            break
    if target_side is None:
        target_side = choose_second_question_target_side(pieces or [])
    if rect_points is None:
        rect_points = second_question_target_rect(target_side).round().astype(np.int32).tolist()

    rect = np.asarray(rect_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(frame, [rect], -1, CYAN, 1)

    for piece in pieces or []:
        if "target_center" not in piece:
            continue
        color = piece_display_color(piece)
        target_polygon = piece.get("target_detected_polygon")
        if target_polygon:
            detected_polygon = np.asarray(target_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(frame, [detected_polygon], -1, color, 2)
        tx, ty = piece["target_center"]
        cv2.circle(frame, (int(tx), int(ty)), 3, color, -1)
        label = piece.get("template", "P%d" % piece.get("id", 0))
        cv2.putText(frame, label, (int(tx) + 4, max(12, int(ty) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


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
        color = piece_display_color(piece)
        target_polygon = piece.get("target_detected_polygon")
        if target_polygon:
            detected_polygon = np.asarray(target_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(frame, [detected_polygon], -1, color, 2)
        tx, ty = piece["target_center"]
        cv2.circle(frame, (int(tx), int(ty)), 3, color, -1)
        if piece.get("template"):
            cv2.putText(frame, piece["template"], (int(tx) + 4, max(12, int(ty) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_piece_debug_info(frame, result, fps, piece_debug):
    draw_text_bg(frame, 8, 24, "A4 OK  FPS:%d" % fps, GREEN, 0.7)
    if PIECE_MASK_METHOD == "hsv":
        mask_info = "mode:%d GREEN H:%d-%d S>=%d" % (
            DISPLAY_MODE,
            PIECE_GREEN_H_LOW,
            PIECE_GREEN_H_HIGH,
            PIECE_GREEN_S_LOW,
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
        detail = "green_bg:HS med:%s/%d" % (
            "on" if PIECE_HSV_MEDIAN_ENABLED else "off",
            PIECE_MASK_MEDIAN_KERNEL if PIECE_HSV_MEDIAN_ENABLED else 0,
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


def make_display_view(frame, result, fps, a4_warp=None, pieces=None, piece_contours=None, piece_debug=None, texture_source=None):
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
            source_view = texture_source if texture_source is not None else view
            draw_target_texture_preview(view, source_view, pieces or [])
            cv2.drawContours(view, piece_debug.get("accepted_source_contours", []), -1, RED, 1)
            draw_question_targets(view, pieces or [])
            draw_pieces(view, pieces or [], piece_contours or [])
            draw_warp_result(view, result, fps)
        draw_piece_debug_info(view, result, fps, piece_debug)
        return view

    if DISPLAY_MODE == 1 and result.get("status"):
        view = a4_warp.copy() if a4_warp is not None else warp_a4(frame, result["corners"])[0]
        source_view = texture_source if texture_source is not None else view
        draw_target_texture_preview(view, source_view, pieces or [])
        draw_question_targets(view, pieces or [])
        if PIECE_DEBUG_OVERLAY_RAW_CONTOUR:
            # 红线 3 像素打底，蓝色最终多边形随后覆盖；重合时仍能看到红色边缘。
            cv2.drawContours(view, piece_debug.get("accepted_source_contours", []), -1, RED, 3)
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
        print("串口打开失败: %s" % err)
        return None


# =========================
# Main Loop
# =========================
# MaixCAM 主循环：采集图像、检测 A4、识别碎片、按键触发稳定捕获、
# 生成移动计划并通过串口发送给下位机。
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
    held_piece_debug = make_empty_piece_debug()
    held_matrix = None
    held_a4_warp = None
    held_until_ms = 0
    capture_wait_logged = False
    if AUTO_CAPTURE_ON_START:
        print("自动捕获已启用")

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
                if held_result is not None:
                    held_result = None
                    held_piece_contours = []
                    held_piece_debug = make_empty_piece_debug()
                    held_matrix = None
                    held_a4_warp = None
                    held_until_ms = 0
                    release_task_memory()
                capture_remaining = CAPTURE_FRAME_COUNT
                capture_samples = []
                capture_wait_logged = False
                print("捕获开始: 帧数=%d" % CAPTURE_FRAME_COUNT)

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
                        # 结果保持期间继续使用最后一个有效采集帧的红色原始轮廓。
                        held_piece_debug = piece_debug
                        held_matrix, _ = a4_perspective_matrices(held_result["corners"])
                        held_a4_warp = cv2.warpPerspective(frame, held_matrix, (WARP_W, WARP_H))
                        held_until_ms = now_for_capture + CAPTURE_HOLD_MS
                        output_result = held_result
                        pieces = held_result.get("pieces", [])
                        piece_contours = held_piece_contours
                        print("捕获成功: 有效帧=%d 碎片数=%d A4=%s" % (
                            held_result.get("capture_frames", 0),
                            len(pieces),
                            held_result.get("a4_method", "unknown"),
                        ))
                        print("A4 STABLE CORNERS TL/TR/BR/BL=%s" % held_result.get("corners"))
                        if MECH_DIRECT_OUTPUT_CALIBRATION_ENABLED:
                            print("MECH OUTPUT CAL TL/TR/BR/BL=%s" % (
                                MECH_OUTPUT_CALIBRATION_POINTS,))
                        else:
                            print("MECH CAL POINTS TL/TR/BR/BL=%s swap_xy=%s" % (
                                MECH_CALIBRATION_POINTS, MECH_SWAP_XY_FOR_STM32))
                        send_move_packet(held_result, serial_obj)
                        released_samples, released_cache = release_task_memory(capture_samples)
                        print("TASK MEMORY RELEASE samples=%d rotation_cache=%d" % (
                            released_samples, released_cache))
                    else:
                        held_result = None
                        held_piece_contours = []
                        held_piece_debug = make_empty_piece_debug()
                        held_matrix = None
                        held_a4_warp = None
                        held_until_ms = 0
                        print("捕获失败: 有效帧=%d" % len(capture_samples))
                        corner_samples = [
                            sample for sample in capture_samples
                            if sample.get("status") and sample.get("corners") is not None
                        ]
                        if corner_samples:
                            print("A4 CAPTURE CORNERS TL/TR/BR/BL=%s" % (
                                average_a4_corners(corner_samples),))
                        released_samples, released_cache = release_task_memory(capture_samples)
                        print("TASK MEMORY RELEASE samples=%d rotation_cache=%d" % (
                            released_samples, released_cache))
            elif held_result is not None and now_for_capture <= held_until_ms:
                output_result = held_result
                pieces = held_result.get("pieces", [])
                piece_contours = held_piece_contours
            elif held_result is not None and now_for_capture > held_until_ms:
                held_result = None
                held_piece_contours = []
                held_piece_debug = make_empty_piece_debug()
                held_matrix = None
                held_a4_warp = None
                held_until_ms = 0
                release_task_memory()
            pieces_end_ms = time.ticks_ms()
        else:
            now_for_capture = time.ticks_ms()
            if capture_requested and not capture_wait_logged:
                print("等待检测A4纸")
                capture_wait_logged = True
            if held_result is not None and now_for_capture <= held_until_ms:
                output_result = held_result
                pieces = held_result.get("pieces", [])
                piece_contours = held_piece_contours
            elif held_result is not None and now_for_capture > held_until_ms:
                held_result = None
                held_piece_contours = []
                held_piece_debug = make_empty_piece_debug()
                held_matrix = None
                held_a4_warp = None
                held_until_ms = 0
                release_task_memory()
            matrix_start_ms = matrix_end_ms = time.ticks_ms()
            pieces_start_ms = pieces_end_ms = matrix_end_ms

        frame_count += 1
        display_start_ms = time.ticks_ms()
        display_result = result
        display_matrix = matrix
        texture_source = None
        if output_result is held_result and held_matrix is not None:
            display_result = held_result
            display_matrix = held_matrix
            piece_debug = held_piece_debug
            texture_source = held_a4_warp
        elif output_result is not result and result.get("status"):
            display_result = result.copy()
            display_result["pieces_count"] = output_result.get("pieces_count", 0)
            display_result["pieces"] = output_result.get("pieces", [])
            display_result["move_plan"] = output_result.get("move_plan", [])
        if a4_warp is None and display_matrix is not None and display_requires_a4_warp(display_result):
            a4_warp = cv2.warpPerspective(frame, display_matrix, (WARP_W, WARP_H))
        view = make_display_view(frame, display_result, fps, a4_warp, pieces, piece_contours, piece_debug, texture_source)
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
                    "性能: n=%d 图像转换=%.1f A4检测=%.1f 矩阵=%.1f 碎片=%.1f 显示=%.1f 总计=%.1f 毫秒"
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
    detector.debug.clear()
    release_task_memory(capture_samples)
    close_runtime_resource(serial_obj)
    close_runtime_resource(cam)
    close_runtime_resource(disp)


if __name__ == "__main__":
    main()
