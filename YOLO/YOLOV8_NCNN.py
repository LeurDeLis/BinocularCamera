# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera 
@File    ：YOLOV8_NCNN.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2025/3/5 下午2:05 
"""

import cv2
from ultralytics import YOLO


class YOLOV8:
    def __init__(self, model_path, confidence=0.65):
        self.model = YOLO(model_path)
        # self.model.to("cpu")
        self.confidence = confidence  # 置信度
        self.max_box = {"square": 0, "x": 0, "y": 0}  # 最大框

    def detect(self, img):
        # transparent_img = np.ones((480, 640, 4), dtype=np.uint8)
        self.max_box["square"], self.max_box["x"], self.max_box["y"], (x, y) = 0, 0, 0, (0, 0)
        label = ""
        # 对帧进行推理，获取检测结果
        results = self.model(img)
        # 将检测结果转换为 NumPy 数组
        results_ = results[0].boxes.data.numpy()
        # 遍历每个检测结果
        for res in results_:
            x1, y1, x2, y2, conf, cls = res  # 解包检测结果
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # 将坐标转换为整数
            (x, y) = (int(x1), int(y1 - 10))
            square = abs((x2 - x1) * (y2 - y1))  # 计算矩形框的面积
            if conf > self.confidence:
                if self.max_box["square"] < square:
                    self.max_box["square"] = square
                    self.max_box["x"], self.max_box["y"] = int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0, 255), 2,
                              lineType=cv2.LINE_AA)  # 绘制矩形框
                # 组合标签内容
                label = f"{self.model.names[cls]}:{conf:.2f}"
        # 返回结果
        return img, self.max_box["x"], self.max_box["y"], (x, y), label

