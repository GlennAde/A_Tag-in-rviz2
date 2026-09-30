import cv2 as cv
import numpy as np
import yaml
from typing import Optional # 容器类别

class camear:
    def __init__(self, camear_id: int = 0, width: int = 640, height: int = 480, fps: int = 30,camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        self.camear_id =  camear_id
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_matrix = camera_matrix  # 内参矩阵
        self.dist_coeffs = dist_coeffs      # 畸变系数

        self.cap = cv.VideoCapture(self.camear_id)
