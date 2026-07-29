import time

import cv2

import config
from a4_detect import A4Detector
from camera import Camera
from utils import draw_a4_result, draw_debug_info, make_process_debug_view


def main():
    camera = Camera()
    if not camera.is_opened():
        print(f"ERROR: cannot open camera {config.CAMERA_ID}")
        return

    detector = A4Detector()
    last_print = 0.0

    print("Stage 1 A4 detection started. Press q or Esc to exit.")
    print(f"Camera: {config.CAMERA_ID}, size: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame")
            break

        result = detector.detect(frame)
        view = frame.copy()
        draw_a4_result(view, result)
        draw_debug_info(view, detector.debug)

        now = time.time()
        if now - last_print >= 0.5:
            print(result)
            last_print = now

        cv2.imshow(config.WINDOW_NAME, view)
        if config.SHOW_PROCESS_WINDOWS:
            debug_view = make_process_debug_view(frame, view, detector.debug)
            cv2.imshow(config.DEBUG_WINDOW_NAME, debug_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
