# 2026 电赛 E 题视觉系统

本仓库用于开发 2026 全国大学生电子设计竞赛 E 题《拼图装置》的视觉系统。

当前完成阶段 1：摄像头输入和 A4 纸外框检测。暂未实现碎片识别、拼图算法、机械控制、ROS 和深度学习。

## 目录结构

```text
.
├── docs/                 # 题目文档
├── maixcam/              # MaixCAM2 运行版本
│   └── main.py
├── opencv/               # 电脑端 OpenCV 调试版本
│   ├── main.py
│   ├── camera.py
│   ├── a4_detect.py
│   ├── config.py
│   └── utils.py
└── test_photo/           # 测试图片
```

## 检测流程

```text
摄像头输入
-> 灰度化
-> 高斯滤波
-> Canny 边缘检测
-> 闭运算连接断边
-> 轮廓查找
-> 四边形拟合
-> A4 面积 / 比例 / 角度筛选
-> 输出四角点
```

检测成功输出：

```python
{"status": True, "corners": [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]}
```

检测失败输出：

```python
{"status": False}
```

## 电脑端运行

依赖：

```bash
pip install opencv-python numpy
```

运行：

```bash
python3 opencv/main.py
```

按 `q` 或 `Esc` 退出。

电脑端会显示两个窗口：

- `Stage 1 - A4 Detection`：检测结果
- `Stage 1 - Process Debug`：原图、灰度、滤波、Canny、闭运算和结果拼图

参数在 `opencv/config.py` 中修改。

## MaixCAM2 运行

MaixCAM2 版本在 `maixcam/main.py`，是单文件入口，方便直接上传到 MaixPy IDE 运行。

板端需要安装 OpenCV：

```bash
pip install opencv-python
```

确认依赖：

```bash
python3 - <<'PY'
import cv2
import numpy as np
print("cv2", cv2.__version__)
print("numpy", np.__version__)
PY
```

注意：MaixCAM2 内置摄像头不能直接用 `cv2.VideoCapture(0)` 打开。本项目使用 MaixPy 摄像头取图：

```python
maix.camera.Camera(...)
image.image2cv(...)
```

然后再交给 OpenCV 处理。

## MaixCAM2 调参

主要参数在 `maixcam/main.py` 顶部：

```python
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_CLOSE_KERNEL = (5, 5)
MORPH_CLOSE_ITERATIONS = 1
APPROX_EPSILON_RATIO = 0.025
MIN_AREA_RATIO = 0.07
MAX_AREA_RATIO = 0.65
CANDIDATE_MIN_RELATIVE_AREA = 0.65
A4_RATIO_MIN = 1.25
A4_RATIO_MAX = 1.65
ANGLE_MIN = 65.0
ANGLE_MAX = 115.0
```

调试显示：

```python
DEBUG_VIEW_MODE = 0  # 原图
DEBUG_VIEW_MODE = 1  # Canny
DEBUG_VIEW_MODE = 2  # 闭运算边缘
```

屏幕调试信息含义：

- `cont`：轮廓数量
- `cand`：最终候选数量
- `pass a/q/r/g`：通过面积、四边形、比例、角度筛选的数量
- `max_area`：最大轮廓面积占比
- `max_pass`：通过面积筛选的最大轮廓面积占比
- `rel_min`：相对面积过滤阈值
- `area`：最终 A4 候选面积占比
- `ratio`：最终 A4 候选长宽比

## 现场建议

- A4 外围黑色边框尽量完整可见。
- 摄像头尽量垂直俯视。
- A4 不要贴满画面，四周留出一定边距。
- 光照尽量均匀，避免强反光。
- 如果离远识别不到，优先降低 `MIN_AREA_RATIO`。
- 如果误识别半张纸，优先提高 `CANDIDATE_MIN_RELATIVE_AREA`。
- 如果边缘断裂，尝试把 `MORPH_CLOSE_KERNEL` 改为 `(7, 7)`。

## 当前阶段

当前版本只解决 A4 检测和四角点输出。A4 检测稳定后，下一阶段再做透视变换，将 A4 转成标准二维坐标系。
