# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera 
@File    ：DistanceMeasurement.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2025/3/5 下午2:04 
"""

import cv2
import numpy as np
from ruamel.yaml import YAML
import math


def get_distance(threeD, x, y):
    """
    普通函数，用于根据像素坐标获取三维坐标和距离。

    参数:
        threeD (numpy.ndarray): 三维点云数据，形状为 (height, width, 3)。
        x (int): 像素坐标的 x 值。
        y (int): 像素坐标的 y 值。

    返回:
        distance: 距离的信息。
    """
    # 打印像素坐标
    print('\n像素坐标 x = %d, y = %d' % (x, y))

    # 获取世界坐标并转换为米
    world_x = threeD[y][x][0] / 1000.0
    world_y = threeD[y][x][1] / 1000.0
    world_z = threeD[y][x][2] / 1000.0
    print("世界坐标xyz 是：", world_x, world_y, world_z, "m")

    # 计算距离并转换为米
    distance = math.sqrt(world_x ** 2 + world_y ** 2 + world_z ** 2)
    print("距离是：", distance, "m")

    # 返回结果
    return distance


class DistanceMeasurement:
    def __init__(self, yaml_file, camera_id=1, image_size=(640, 480)):
        """
        初始化双目视觉类。
        :param yaml_file: 标定参数文件路径（YAML 格式）。
        :param camera_id: 摄像头 ID 或视频文件路径。
        :param image_size: 图像分辨率，默认为 (640, 480)。
        """
        self.Q = None
        self.left_map1, self.left_map2 = None, None
        self.right_map1, self.right_map2 = None, None
        self.yaml_file = yaml_file
        self.camera_id = camera_id
        self.image_size = image_size

        # 加载标定参数
        self.load_calibration_params()

        # 初始化摄像头
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 初始化 SGBM 参数
        self.blockSize = 9  # 初始值为 9，范围在 5-25 之间
        self.num = 15  # 初始值为 15，必须大于 0

        # 创建窗口和滑动条
        self.WIN_NAME = 'Deep disp'
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Block Size", self.WIN_NAME, self.blockSize, 25, self.on_blockSize_change)
        cv2.createTrackbar("Num", self.WIN_NAME, self.num, 50, self.on_num_change)

    def load_calibration_params(self):
        """加载双目相机的标定参数"""
        yaml = YAML()
        with open(self.yaml_file, 'r') as file:
            data = yaml.load(file)

        # 提取数据
        param = data.get("camera_calibration", {})
        # 左镜头的内参，如焦距
        left_camera_matrix = np.array(param['left_camera']['k'])
        right_camera_matrix = np.array(param['right_camera']['k'])
        # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
        left_distortion = np.array(param['left_camera']['d'])
        right_distortion = np.array(param['right_camera']['d'])
        # 旋转矩阵
        R = np.array(param['extrinsic_parameters']['R'])
        # 平移矩阵
        T = np.array(param['extrinsic_parameters']['T'])

        # 计算校正参数
        # R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
        R1, R2, P1, P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                               right_camera_matrix, right_distortion,
                                                                               self.image_size, R, T)

        # 计算校正映射表
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1,
                                                                     self.image_size, cv2.CV_32FC1)

        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2,
                                                                       self.image_size, cv2.CV_32FC1)

    def on_blockSize_change(self, value):
        """滑动条回调函数，调整 blockSize"""
        self.blockSize = max(5, min(25, value))

    def on_num_change(self, value):
        """滑动条回调函数，调整 num"""
        self.num = max(1, value)

    def compute_disparity(self, imgL, imgR):
        """计算视差图"""
        stereo = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=self.num * 16,
            blockSize=self.blockSize,
            P1=8 * 3 * self.blockSize * self.blockSize,
            P2=32 * 3 * self.blockSize * self.blockSize,
            disp12MaxDiff=-1,
            preFilterCap=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=100,
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def process_stereo_image(self, frame):
        """
        处理双目图像以生成视差图、深度图和三维坐标数据。

        参数:
            frame (numpy.ndarray): 输入的双目图像帧（左右图像拼接在一起）。
        返回:
            - 'frame_l': 视差图（未经归一化）。
            - 'depth_img': 深度图（灰度图）。
            - 'depth_color': 深度图（颜色图）。
            - 'threeD': 三维坐标数据（单位：毫米）。
        """
        # 分割左右图像
        frame_l = frame[0:480, 0:640]
        frame_r = frame[0:480, 640:1280]

        # 转换为灰度图像
        imgL = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # 校正图像
        img1_rectified = cv2.remap(imgL, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        # 计算视差图
        disparity = self.compute_disparity(img1_rectified, img2_rectified)

        # 归一化生成深度图（灰度图）
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 归一化生成深度图（颜色图）
        dis_color = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color = cv2.applyColorMap(dis_color, cv2.COLORMAP_JET)

        # 计算三维坐标数据值
        threeD = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)
        threeD = threeD * 16  # 单位转换为毫米

        # 返回结果
        return frame_l, disp, dis_color, threeD
