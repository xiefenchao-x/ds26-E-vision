#!/usr/bin/env python3
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = "115200"
DEFAULT_FEEDRATE = "3000"
DEFAULT_PEN_UP_Z = "5"
DEFAULT_PEN_DOWN_Z = "0"


class WriterDebugApp:
    def __init__(self, root):
        self.root = root
        self.root.title("写字机 G-code 调试工具")
        self.root.geometry("980x640")
        self.root.minsize(860, 560)

        self.serial_obj = None
        self.rx_queue = queue.Queue()
        self.reader_stop = threading.Event()
        self.reader_thread = None

        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        self.baudrate_var = tk.StringVar(value=DEFAULT_BAUDRATE)
        self.feedrate_var = tk.StringVar(value=DEFAULT_FEEDRATE)
        self.pen_up_z_var = tk.StringVar(value=DEFAULT_PEN_UP_Z)
        self.pen_down_z_var = tk.StringVar(value=DEFAULT_PEN_DOWN_Z)
        self.x_var = tk.StringVar(value="10")
        self.y_var = tk.StringVar(value="10")
        self.jog_var = tk.StringVar(value="10")
        self.box_x_var = tk.StringVar(value="10")
        self.box_y_var = tk.StringVar(value="10")
        self.box_w_var = tk.StringVar(value="20")
        self.box_h_var = tk.StringVar(value="20")
        self.raw_var = tk.StringVar(value="G0 X10 Y10 F3000")
        self.absolute_var = tk.BooleanVar(value=True)
        self.units_mm_var = tk.BooleanVar(value=True)
        self.rapid_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="未连接")

        self.build_ui()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self.poll_rx_queue)

    def build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="ns")

        right = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self.build_serial_panel(left)
        self.build_motion_panel(left)
        self.build_pen_panel(left)
        self.build_box_panel(left)
        self.build_raw_panel(right)
        self.build_log_panel(right)

    def build_serial_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="串口", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="端口").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(frame, textvariable=self.port_var, width=18)
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="刷新", command=self.refresh_ports).grid(row=0, column=2)

        ttk.Label(frame, text="波特率").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.baudrate_var, width=12).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))

        row = ttk.Frame(frame)
        row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.connect_button = ttk.Button(row, text="连接", command=self.toggle_connection)
        self.connect_button.pack(side="left")
        ttk.Button(row, text="清空日志", command=self.clear_log).pack(side="left", padx=6)

        ttk.Label(frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def build_motion_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="移动", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        ttk.Label(grid, text="X").grid(row=0, column=0)
        ttk.Entry(grid, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=(4, 10))
        ttk.Label(grid, text="Y").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=(4, 10))
        ttk.Label(grid, text="F").grid(row=0, column=4)
        ttk.Entry(grid, textvariable=self.feedrate_var, width=8).grid(row=0, column=5, padx=(4, 0))

        options = ttk.Frame(frame)
        options.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(options, text="绝对坐标 G90", variable=self.absolute_var).pack(side="left")
        ttk.Checkbutton(options, text="毫米 G21", variable=self.units_mm_var).pack(side="left", padx=8)
        ttk.Checkbutton(options, text="快速 G0", variable=self.rapid_var).pack(side="left")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="移动到 XY", command=self.move_to_xy).pack(side="left")
        ttk.Button(row, text="回零 $H", command=lambda: self.send_gcode("$H")).pack(side="left", padx=6)
        ttk.Button(row, text="解锁 $X", command=lambda: self.send_gcode("$X")).pack(side="left")

        jog = ttk.Frame(frame)
        jog.pack(pady=(10, 0))
        ttk.Label(jog, text="步长").grid(row=0, column=0)
        ttk.Entry(jog, textvariable=self.jog_var, width=8).grid(row=0, column=1, padx=4)
        ttk.Button(jog, text="Y+", width=8, command=lambda: self.jog(0, 1)).grid(row=0, column=2, padx=4)
        ttk.Button(jog, text="X-", width=8, command=lambda: self.jog(-1, 0)).grid(row=1, column=1, pady=4)
        ttk.Button(jog, text="X+", width=8, command=lambda: self.jog(1, 0)).grid(row=1, column=3, pady=4)
        ttk.Button(jog, text="Y-", width=8, command=lambda: self.jog(0, -1)).grid(row=2, column=2, padx=4)

    def build_pen_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="笔", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="抬笔 Z").pack(side="left")
        ttk.Entry(row, textvariable=self.pen_up_z_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="落笔 Z").pack(side="left")
        ttk.Entry(row, textvariable=self.pen_down_z_var, width=8).pack(side="left", padx=4)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="抬笔", command=self.pen_up).pack(side="left")
        ttk.Button(buttons, text="落笔", command=self.pen_down).pack(side="left", padx=6)

    def build_box_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="测试方框", padding=8)
        frame.pack(fill="x")

        line1 = ttk.Frame(frame)
        line1.pack(fill="x")
        ttk.Label(line1, text="X").pack(side="left")
        ttk.Entry(line1, textvariable=self.box_x_var, width=7).pack(side="left", padx=(4, 8))
        ttk.Label(line1, text="Y").pack(side="left")
        ttk.Entry(line1, textvariable=self.box_y_var, width=7).pack(side="left", padx=(4, 8))
        ttk.Label(line1, text="W").pack(side="left")
        ttk.Entry(line1, textvariable=self.box_w_var, width=7).pack(side="left", padx=(4, 8))
        ttk.Label(line1, text="H").pack(side="left")
        ttk.Entry(line1, textvariable=self.box_h_var, width=7).pack(side="left", padx=(4, 0))

        ttk.Button(frame, text="画方框", command=self.draw_box).pack(anchor="w", pady=(8, 0))

    def build_raw_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="手动 G-code", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        entry = ttk.Entry(frame, textvariable=self.raw_var)
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<Return>", lambda _event: self.send_raw())
        ttk.Button(frame, text="发送", command=self.send_raw).grid(row=0, column=1, padx=(8, 0))

        quick = ttk.Frame(frame)
        quick.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        for text, command in (
            ("G90", "G90"),
            ("G91", "G91"),
            ("G21", "G21"),
            ("M3 S1000", "M3 S1000"),
            ("M5", "M5"),
            ("?", "?"),
        ):
            ttk.Button(quick, text=text, command=lambda value=command: self.send_gcode(value)).pack(side="left", padx=(0, 6))

    def build_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="日志", padding=8)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(frame, wrap="word", height=12)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def refresh_ports(self):
        ports = []
        try:
            from serial.tools import list_ports

            ports = [port.device for port in list_ports.comports()]
        except Exception:
            ports = []

        if not ports:
            ports = [DEFAULT_PORT, "/dev/ttyACM0", "COM3"]
        self.port_combo.configure(values=ports)
        if self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def toggle_connection(self):
        if self.serial_obj is None:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        try:
            import serial

            self.serial_obj = serial.Serial(
                port=self.port_var.get(),
                baudrate=int(self.baudrate_var.get()),
                timeout=0.05,
                write_timeout=0.5,
            )
            time.sleep(2.0)
            self.serial_obj.reset_input_buffer()
        except Exception as err:
            self.serial_obj = None
            messagebox.showerror("连接失败", str(err))
            return

        self.reader_stop.clear()
        self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.reader_thread.start()
        self.connect_button.configure(text="断开")
        self.status_var.set("已连接 %s @ %s" % (self.port_var.get(), self.baudrate_var.get()))
        self.log("OPEN %s @ %s" % (self.port_var.get(), self.baudrate_var.get()))

    def disconnect(self):
        self.reader_stop.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=0.5)
            self.reader_thread = None
        if self.serial_obj is not None:
            try:
                self.serial_obj.close()
            except Exception:
                pass
            self.serial_obj = None
        self.connect_button.configure(text="连接")
        self.status_var.set("未连接")
        self.log("CLOSE")

    def reader_loop(self):
        while not self.reader_stop.is_set():
            try:
                if self.serial_obj is not None and self.serial_obj.in_waiting:
                    data = self.serial_obj.read(self.serial_obj.in_waiting)
                    if data:
                        self.rx_queue.put(data.decode("utf-8", errors="replace"))
            except Exception as err:
                self.rx_queue.put("RX ERROR: %s\n" % err)
                break
            time.sleep(0.03)

    def poll_rx_queue(self):
        while True:
            try:
                text = self.rx_queue.get_nowait()
            except queue.Empty:
                break
            self.log("RX: " + text.rstrip())
        self.root.after(80, self.poll_rx_queue)

    def send_gcode(self, line):
        line = line.strip()
        if not line:
            return
        self.log("TX: " + line)
        if self.serial_obj is None:
            self.log("WARN: 串口未连接，只显示未发送")
            return
        try:
            self.serial_obj.write((line + "\n").encode("ascii"))
            self.serial_obj.flush()
        except Exception as err:
            messagebox.showerror("发送失败", str(err))

    def send_preface(self):
        if self.absolute_var.get():
            self.send_gcode("G90")
        if self.units_mm_var.get():
            self.send_gcode("G21")

    def send_raw(self):
        self.send_gcode(self.raw_var.get())

    def move_to_xy(self):
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            feedrate = float(self.feedrate_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "X、Y、F 必须是数字")
            return
        self.send_preface()
        code = "G0" if self.rapid_var.get() else "G1"
        self.send_gcode("%s X%s Y%s F%s" % (code, fmt_number(x), fmt_number(y), fmt_number(feedrate)))

    def jog(self, dx, dy):
        try:
            step = float(self.jog_var.get())
            feedrate = float(self.feedrate_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "步长和 F 必须是数字")
            return
        self.send_gcode("G91")
        self.send_gcode("G1 X%s Y%s F%s" % (fmt_number(dx * step), fmt_number(dy * step), fmt_number(feedrate)))
        if self.absolute_var.get():
            self.send_gcode("G90")

    def pen_up(self):
        self.move_z(self.pen_up_z_var.get())

    def pen_down(self):
        self.move_z(self.pen_down_z_var.get())

    def move_z(self, z_value):
        try:
            z = float(z_value)
            feedrate = float(self.feedrate_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "Z 和 F 必须是数字")
            return
        self.send_preface()
        self.send_gcode("G1 Z%s F%s" % (fmt_number(z), fmt_number(feedrate)))

    def draw_box(self):
        try:
            x = float(self.box_x_var.get())
            y = float(self.box_y_var.get())
            w = float(self.box_w_var.get())
            h = float(self.box_h_var.get())
            up_z = float(self.pen_up_z_var.get())
            down_z = float(self.pen_down_z_var.get())
            feedrate = float(self.feedrate_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "方框、Z、F 参数必须是数字")
            return

        self.send_preface()
        for line in (
            "G1 Z%s F%s" % (fmt_number(up_z), fmt_number(feedrate)),
            "G1 X%s Y%s F%s" % (fmt_number(x), fmt_number(y), fmt_number(feedrate)),
            "G1 Z%s F%s" % (fmt_number(down_z), fmt_number(feedrate)),
            "G1 X%s Y%s F%s" % (fmt_number(x + w), fmt_number(y), fmt_number(feedrate)),
            "G1 X%s Y%s F%s" % (fmt_number(x + w), fmt_number(y + h), fmt_number(feedrate)),
            "G1 X%s Y%s F%s" % (fmt_number(x), fmt_number(y + h), fmt_number(feedrate)),
            "G1 X%s Y%s F%s" % (fmt_number(x), fmt_number(y), fmt_number(feedrate)),
            "G1 Z%s F%s" % (fmt_number(up_z), fmt_number(feedrate)),
        ):
            self.send_gcode(line)

    def log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", "[%s] %s\n" % (timestamp, text))
        self.log_text.see("end")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def close(self):
        self.disconnect()
        self.root.destroy()


def fmt_number(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return ("%.3f" % number).rstrip("0").rstrip(".")


def main():
    root = tk.Tk()
    WriterDebugApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
