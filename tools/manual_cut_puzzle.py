#!/usr/bin/env python3
import math
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import cv2
import numpy as np


CANVAS_W = 760
CANVAS_H = 520
RECT_X0 = 70
RECT_Y0 = 70
RECT_W = 620
RECT_H = 360
CUT_THICKNESS = 3
MIN_AREA = 400
SNAP_DISTANCE = 14
BOUNDARY_EXTEND = 12

BG = "#25282b"
RECT_FILL = "#f7f7f7"
RECT_BORDER = "#ffffff"
CUT_COLOR = "#f0c400"
PIECE_COLOR = "#00d2d2"


class ManualCutPuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手动切割拼图")
        self.root.geometry("880x610")
        self.root.minsize(820, 560)

        self.points = []
        self.segments = []
        self.anchor = None
        self.last_point = None
        self.pieces = []
        self.photo = None
        self.status = tk.StringVar(
            value="单击放点；两点自动成线。按“加锚点”后，后续点都会从锚点发出。"
        )

        self.build_ui()
        self.redraw()

    def build_ui(self):
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0, width=CANVAS_W, height=CANVAS_H)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self.canvas.bind("<Button-1>", self.on_click)

        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=(0, 6))
        buttons = [
            ("撤销一刀", self.undo),
            ("重置", self.reset),
            ("加锚点", self.set_anchor),
            ("清空线", self.clear_lines),
            ("执行切割", self.execute_cut),
            ("取消锚点", self.clear_anchor),
        ]
        for text, command in buttons:
            ttk.Button(row, text=text, command=command).pack(side="left", padx=(0, 8), ipadx=12, ipady=8)

        ttk.Label(self.root, textvariable=self.status, foreground="#174c75").pack(anchor="w", padx=10)

    def on_click(self, event):
        point = self.clamp_to_canvas((event.x, event.y))
        self.points.append(point)
        if self.anchor is not None:
            if self.distance(self.anchor, point) > 2:
                self.segments.append((self.anchor, point))
            self.last_point = point
        elif self.last_point is None:
            self.last_point = point
        else:
            if self.distance(self.last_point, point) > 2:
                self.segments.append((self.last_point, point))
            self.last_point = point
        self.pieces = []
        self.redraw()

    def set_anchor(self):
        if self.last_point is None:
            messagebox.showinfo("没有锚点", "先在画布上点一下内部交汇点。")
            return
        self.anchor = self.last_point
        self.status.set("已设置锚点：后续每个点都会和锚点连成一刀。")
        self.redraw()

    def clear_anchor(self):
        self.anchor = None
        self.status.set("已取消锚点：恢复为两点自动成线。")
        self.redraw()

    def undo(self):
        if self.segments:
            self.segments.pop()
        if self.points:
            self.points.pop()
        self.last_point = self.points[-1] if self.points else None
        if self.anchor is not None and self.anchor not in self.points:
            self.anchor = None
        self.pieces = []
        self.redraw()

    def reset(self):
        self.points = []
        self.segments = []
        self.anchor = None
        self.last_point = None
        self.pieces = []
        self.status.set("已重置。")
        self.redraw()

    def clear_lines(self):
        self.segments = []
        self.pieces = []
        self.status.set("已清空切割线，点和锚点保留。")
        self.redraw()

    def execute_cut(self):
        pieces = self.cut_regions()
        if len(pieces) < 2:
            self.status.set("这些线还没有切开矩形；线段需要从边界到边界，或形成连接到边界的切割图。")
            return
        self.pieces = pieces
        edge_counts = ", ".join("P%d:%d边" % (idx, len(piece)) for idx, piece in enumerate(pieces))
        self.status.set("切割完成：%d 块；%s" % (len(pieces), edge_counts))
        self.redraw()

    def cut_regions(self):
        mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        cv2.rectangle(mask, (RECT_X0, RECT_Y0), (RECT_X0 + RECT_W, RECT_Y0 + RECT_H), 255, -1)
        for start, end in self.segments:
            cut_start, cut_end = self.segment_for_cut(start, end)
            cv2.line(mask, self.int_point(cut_start), self.int_point(cut_end), 0, CUT_THICKNESS, lineType=cv2.LINE_8)

        count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, 8)
        pieces = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.01), True)
            polygon = approx.reshape(-1, 2)
            polygon = self.drop_near_duplicate_points(polygon)
            if len(polygon) >= 3:
                pieces.append(polygon)
        pieces.sort(key=lambda poly: (float(np.mean(poly[:, 1])), float(np.mean(poly[:, 0]))))
        return pieces

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=BG, outline="")
        self.canvas.create_rectangle(
            RECT_X0, RECT_Y0, RECT_X0 + RECT_W, RECT_Y0 + RECT_H,
            fill=RECT_FILL, outline=RECT_BORDER, width=2,
        )

        if self.pieces:
            for idx, polygon in enumerate(self.pieces):
                flat = polygon.astype(int).flatten().tolist()
                self.canvas.create_polygon(flat, outline=PIECE_COLOR, fill="", width=2)
                center = np.mean(polygon, axis=0)
                self.canvas.create_text(
                    int(center[0]), int(center[1]),
                    text="P%d" % idx,
                    fill="#303030",
                    font=("Sans", 18, "bold"),
                )

        for start, end in self.segments:
            self.canvas.create_line(*start, *end, fill=CUT_COLOR, width=2)
        for idx, point in enumerate(self.points):
            fill = "#ffe36a" if point == self.anchor else CUT_COLOR
            self.canvas.create_oval(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5, fill=fill, outline="#111")
            self.canvas.create_text(point[0] + 12, point[1] - 8, text="A" if point == self.anchor else str(idx), fill=CUT_COLOR)

    @staticmethod
    def drop_near_duplicate_points(points):
        result = []
        for point in points:
            if not result or np.linalg.norm(point - result[-1]) > 2.0:
                result.append(point)
        if len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= 2.0:
            result.pop()
        return np.asarray(result, dtype=np.int32)

    @staticmethod
    def clamp_to_canvas(point):
        x, y = point
        return (max(0, min(CANVAS_W - 1, int(x))), max(0, min(CANVAS_H - 1, int(y))))

    @staticmethod
    def snap_to_rect(point):
        x, y = float(point[0]), float(point[1])
        inside_y = RECT_Y0 - SNAP_DISTANCE <= y <= RECT_Y0 + RECT_H + SNAP_DISTANCE
        inside_x = RECT_X0 - SNAP_DISTANCE <= x <= RECT_X0 + RECT_W + SNAP_DISTANCE
        candidates = []
        if inside_y:
            candidates.append((abs(x - RECT_X0), (RECT_X0, y)))
            candidates.append((abs(x - (RECT_X0 + RECT_W)), (RECT_X0 + RECT_W, y)))
        if inside_x:
            candidates.append((abs(y - RECT_Y0), (x, RECT_Y0)))
            candidates.append((abs(y - (RECT_Y0 + RECT_H)), (x, RECT_Y0 + RECT_H)))
        candidates = [item for item in candidates if item[0] <= SNAP_DISTANCE]
        if not candidates:
            return (x, y)
        _distance, snapped = min(candidates, key=lambda item: item[0])
        return snapped

    @staticmethod
    def outward_vector_for_boundary(point):
        x, y = float(point[0]), float(point[1])
        if abs(x - RECT_X0) <= 1.5:
            return (-1.0, 0.0)
        if abs(x - (RECT_X0 + RECT_W)) <= 1.5:
            return (1.0, 0.0)
        if abs(y - RECT_Y0) <= 1.5:
            return (0.0, -1.0)
        if abs(y - (RECT_Y0 + RECT_H)) <= 1.5:
            return (0.0, 1.0)
        return None

    @classmethod
    def segment_for_cut(cls, start, end):
        start = cls.snap_to_rect(start)
        end = cls.snap_to_rect(end)
        start_out = cls.outward_vector_for_boundary(start)
        end_out = cls.outward_vector_for_boundary(end)
        if start_out is not None:
            start = (
                float(start[0]) + start_out[0] * BOUNDARY_EXTEND,
                float(start[1]) + start_out[1] * BOUNDARY_EXTEND,
            )
        if end_out is not None:
            end = (
                float(end[0]) + end_out[0] * BOUNDARY_EXTEND,
                float(end[1]) + end_out[1] * BOUNDARY_EXTEND,
            )
        return start, end

    @staticmethod
    def int_point(point):
        return (int(round(point[0])), int(round(point[1])))

    @staticmethod
    def distance(a, b):
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def main():
    root = tk.Tk()
    app = ManualCutPuzzleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
