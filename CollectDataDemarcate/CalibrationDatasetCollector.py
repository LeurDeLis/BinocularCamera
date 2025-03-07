# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera 
@File    ：CalibrationDatasetCollector.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2025/2/27 下午2:09 
"""

import cv2

# 获取视频文件路径或使用默认摄像头
video_file = 1

# 打开视频流
cap = cv2.VideoCapture(video_file)

cap.set(3, 1280)
cap.set(4, 480)  # 打开并设置摄像头

if not cap.isOpened():
    print("摄像头打开失败!")
    exit(-1)

count1 = 0
count2 = 0

cv2.namedWindow("图片1", cv2.WINDOW_NORMAL)
cv2.namedWindow("图片2", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()  # 从摄像头捕获一帧图像
    if not ret:
        print("无法读取视频帧!")
        break

    frame_L = frame[0:480, 0:640]
    frame_R = frame[0:480, 640:1280]  # 割开双目图像

    # 显示左右图像
    cv2.imshow("Video_L", frame_L)
    cv2.imshow("Video_R", frame_R)

    key = cv2.waitKey(50) & 0xFF  # 等待按键输入

    if key == 27:  # 按下ESC键退出
        break

    if key == 32:  # 按下空格键保存图像
        count1 += 1
        count2 += 1

        image_left = f"./data/left/left_{count1}.jpg"
        image_right = f"./data/right/right_{count2}.jpg"

        print(f"保存第{count1}张图片")

        cv2.imwrite(image_left, frame_L)
        cv2.imwrite(image_right, frame_R)

        cv2.imshow("图片1", frame_L)
        cv2.imshow("图片2", frame_R)

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
