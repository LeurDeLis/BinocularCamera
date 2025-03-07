# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2025/3/5 下午1:13 
"""

import time
import cv2
from YOLOV8_NCNN import YOLOV8
from DistanceMeasurement import DistanceMeasurement, get_distance


if __name__ == '__main__':
    # 初始化系统：加载双目相机参数、YOLO模型和摄像头
    param_file = '../CameraParam/StereoParameters_01.yaml'
    stereo_vision = DistanceMeasurement(param_file, camera_id=1)

    onnx_path = './weights/best_prune_ncnn_model'  # 模型路径
    model = YOLOV8(onnx_path)  # 创建YOLOv8模型

    cap = cv2.VideoCapture(1)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        start_time = time.time()

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出程序。")
            break

        # 双目视觉处理
        frame_l, disp, dis_color, threeD = stereo_vision.process_stereo_image(frame)

        # 目标检测
        frame_l, x, y, (txt_x, txt_y), label = model.detect(frame_l)

        # 距离计算
        distance = get_distance(threeD, x, y)
        label_with_distance = f"{label} {distance:.2f}cm"

        # 计算FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # 显示FPS
        cv2.putText(frame_l, f"FPS: {fps:.1f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    thickness=2, lineType=cv2.LINE_AA)

        # 显示标签和距离
        cv2.putText(frame_l, label_with_distance, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255),
                    thickness=2, lineType=cv2.LINE_AA)

        # 显示窗口
        cv2.imshow("Depth Map", dis_color)
        cv2.imshow('Object Detection Result', frame_l)
        cv2.imshow(stereo_vision.WIN_NAME, disp)

        # 检测退出条件
        key = cv2.waitKey(1)
        if key in [27, ord('q'), 32]:  # ESC、'q' 或空格键退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


