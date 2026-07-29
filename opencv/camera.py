import cv2

import config


class Camera:
    def __init__(self, camera_id=config.CAMERA_ID):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
