# 摄像头编号。电脑只有一个摄像头时通常是 0；外接 USB 摄像头可能是 1 或 2。
CAMERA_ID = 0

# 输入分辨率。先用 640x480，速度快、调试方便；如果角点抖动明显，可试 1280x720。
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WINDOW_NAME = "Stage 1 - A4 Detection"
DEBUG_WINDOW_NAME = "Stage 1 - Process Debug"

# 高斯滤波核。越大越抗噪，但边缘会更钝；现场一般用 (5, 5)。
GAUSSIAN_KERNEL = (5, 5)

# Canny 边缘阈值。
# 检不出纸边：降低，例如 30 / 100。
# 杂边太多、误检多：升高，例如 80 / 200。
CANNY_LOW = 50
CANNY_HIGH = 150

# 闭运算用于连接断掉的纸张边缘。
# 纸边断裂：可改成 (7, 7)，并在 a4_detect.py 里把 iterations 改成 2。
# 轮廓粘连：改回 (3, 3) 或 (5, 5)。
MORPH_CLOSE_KERNEL = (9, 9)

# 多边形拟合精度，值越大越容易把轮廓拟合成四边形，但过大会把形状拟歪。
# 推荐范围：0.015 ~ 0.04。
APPROX_EPSILON_RATIO = 0.02

# A4 轮廓面积占整幅图像的比例。
# 现在你的画面里 A4 很大，0.12 ~ 0.90 合适。
# 如果摄像头离得远、A4较小，MIN_AREA_RATIO 可降到 0.05。
MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.98

# A4 长宽比：297 / 210 = 1.414。
# 透视角度越斜，图像里的比例偏差越大，所以这里不要卡太死。
A4_ASPECT = 297.0 / 210.0
A4_RATIO_MIN = 1.20
A4_RATIO_MAX = 1.75

# 四个角接近 90 度的容差。
# 只有某些角度能识别：放宽到 55 / 125。
# 误检其它矩形：收紧到 75 / 105。
ANGLE_MIN = 65.0
ANGLE_MAX = 115.0

# 是否显示左下角调试信息。现场调参建议打开。
SHOW_DEBUG_INFO = True

# 是否显示处理中间过程窗口。
# 打开后会显示：原图、灰度、模糊、Canny、闭运算边缘、最终检测结果。
SHOW_PROCESS_WINDOWS = True

# 调试拼图里每个小图的宽高，调小可以减少屏幕占用。
DEBUG_TILE_WIDTH = 320
DEBUG_TILE_HEIGHT = 240

COLOR_SUCCESS = (0, 220, 0)
COLOR_FAIL = (0, 0, 255)
COLOR_CORNER = (0, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
