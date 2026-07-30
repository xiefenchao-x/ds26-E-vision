import cv2
import numpy as np

import config
from utils import compute_midline, order_points, quad_angles, quad_aspect_ratio


class A4Detector:
    def __init__(self):
        self.debug = {}

    def detect(self, frame):
        self.debug = {}
        frame_area = frame.shape[0] * frame.shape[1]

        # 1. 传统边缘检测：灰度、滤波、Canny、闭运算连接断边。
        gray, blur, canny, edges = self._preprocess(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        best_ratio = 0.0
        best_angles = []
        candidate_count = 0

        for contour in contours:
            # 2. 面积过滤：太小多半是碎片/噪声，太大可能是整幅画面边界。
            area = cv2.contourArea(contour)
            area_ratio = area / frame_area
            if area_ratio < config.MIN_AREA_RATIO or area_ratio > config.MAX_AREA_RATIO:
                continue

            # 3. 多边形拟合：只保留凸四边形。
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON_RATIO * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            # 4. 角点排序后计算长宽比。输出顺序固定为 TL, TR, BR, BL。
            corners = order_points(approx.reshape(4, 2))
            aspect_ratio = quad_aspect_ratio(corners)
            if not (config.A4_RATIO_MIN <= aspect_ratio <= config.A4_RATIO_MAX):
                continue

            # 5. 四角角度过滤。摄像头越斜，角度越偏离 90 度。
            angles = quad_angles(corners)
            if not all(config.ANGLE_MIN <= angle <= config.ANGLE_MAX for angle in angles):
                continue

            # 6. 多个候选同时满足时，取面积最大的那个作为 A4。
            candidate_count += 1
            if area > best_area:
                best_area = area
                best_ratio = aspect_ratio
                best_angles = angles
                best = corners

        self.debug = {
            "gray": gray,
            "blur": blur,
            "canny": canny,
            "edges": edges,
            "contours": len(contours),
            "candidates": candidate_count,
            "best_area": best_area,
            "best_area_ratio": best_area / frame_area if frame_area > 0 else 0.0,
            "best_aspect_ratio": best_ratio,
            "best_angles": best_angles,
        }

        if best is None:
            return {"status": False}

        corners = best.astype(int).tolist()
        return {
            "status": True,
            "corners": corners,
            "midline": compute_midline(corners),
        }

    @staticmethod
    def _preprocess(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, 0)
        canny = cv2.Canny(blur, config.CANNY_LOW, config.CANNY_HIGH)
        kernel = np.ones(config.MORPH_CLOSE_KERNEL, np.uint8)
        edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)
        return gray, blur, canny, edges
