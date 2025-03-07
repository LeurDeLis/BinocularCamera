# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera
@File    ：CameraDemarcate.py
@IDE     ：PyCharm
@Author  ：MFK
@Date    ：2025/3/3 下午4:16
"""

import os
import numpy as np
import glob
import cv2
from ruamel.yaml import YAML


def generate_coordinate_grid(shape, axis=0):
    """
    生成一个二维坐标网格矩阵，用于表示图像中某一方向上的坐标索引。
    :param shape: 图像的形状，格式为 (height, width)，即图像的高度和宽度。
    :param axis: 指定生成坐标的轴方向。
                - 如果 axis=0，则生成沿水平方向（x 轴）变化的坐标网格。
                - 如果 axis=1，则生成沿垂直方向（y 轴）变化的坐标网格。

    :return: numpy.ndarray: 一个形状为 (height, width) 的二维数组，表示指定方向上的坐标索引。
                       - 当 axis=0 时，每一行的值相同，表示水平方向的 x 坐标。
                       - 当 axis=1 时，每一列的值相同，表示垂直方向的 y 坐标。
   """
    h, w = shape
    if axis == 0:
        coordinate = np.reshape(np.array(range(w)), [1, w])
        coordinate = coordinate.repeat(h, axis=axis)
    else:
        coordinate = np.reshape(np.array(range(h)), [h, 1])
        coordinate = coordinate.repeat(w, axis=axis)
    return coordinate.reshape(h, w)


def rectify_image(image, K, D, R, P, interpolation=cv2.INTER_LINEAR):
    """
    对图像进行去畸变和校正。
    :param image: 输入图像。
    :param K: 内参矩阵。
    :param D: 畸变系数。
    :param R: 旋转变换矩阵。
    :param P: 投影矩阵。
    :param interpolation: 插值方法。
    :return: 校正后的图像。
    """
    h, w = image.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, R, P, (w, h), cv2.CV_32FC1)
    image_Rectify = cv2.remap(image, map_x, map_y, interpolation)

    return image_Rectify


class StereoCameraDemarcate(object):
    def __init__(self,
                 data_root,
                 imgL_root=None,
                 imgR_root=None,
                 square_size=20,
                 corner_size=(9, 6),
                 suffix="jpg",
                 image_shape=(480, 1280),
                 camera_param_file="StereoParameters"):
        """
        初始化双目相机标定器类。
        :param data_root: 数据根目录路径。
        :param imgL_root: 左相机图像路径（可选）。
        :param imgR_root: 右相机图像路径（可选）。
        :param square_size: 标定板方格的实际物理尺寸（单位：mm）。
        :param corner_size: 标定板角点行列数，默认为 (9, 6)。
        :param suffix: 图像文件后缀名，默认为 "jpg"。
        :param image_shape: 图像尺寸，默认为 (480, 640)。
        :param camera_param_file: 相机参数保存文件名。
        """
        super(StereoCameraDemarcate, self).__init__()
        self.Calib = None
        self.K1, self.K2, self.D1, self.D2, self.R, self.T, self.R1, self.R2, self.P1, self.P2, self.Q = (
            None, None, None, None, None, None, None, None, None, None, None)
        self.data_root = data_root
        self.corner_h, self.corner_w = corner_size
        self.H, self.W = image_shape
        # 标定板方格的实际物理尺寸（单位：mm）
        self.square_size = square_size
        self.img_shape = (self.W, self.H)
        self.yaml_file = os.path.join("../CameraParam/", f"{camera_param_file}.yaml")
        self.suffix = suffix

        # 终止条件, 用于角点检测和标定优化
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # 准备三维空间中的标定点，如 (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
        self.objp = np.zeros((self.corner_h * self.corner_w, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.corner_w, 0:self.corner_h].T.reshape(-1, 2)
        self.objp *= self.square_size
        # 存储标定点和图像点
        self.obj_points = []  # 真实世界中的 3D 点
        self.img_points_l = []  # 左相机图像中的 2D 点
        self.img_points_r = []  # 右相机图像中的 2D 点

        # 设置左右相机图像路径
        if imgL_root is None and imgR_root is None:
            self.imgL_root = os.path.join(self.data_root, 'left/*.{}'.format(suffix))
            self.imgR_root = os.path.join(self.data_root, 'right/*.{}'.format(suffix))
        else:
            self.imgL_root = os.path.join(imgL_root, '*.{}'.format(suffix))
            self.imgR_root = os.path.join(imgR_root, '*.{}'.format(suffix))

    '''读取左右图角点'''
    def read_images(self, winSize=(5, 5)):
        """
        读取左右图像并提取角点。
        :param winSize: 搜索窗口大小。
        """
        print("开始读取图像角点")
        images_left = glob.glob(self.imgL_root)
        images_right = glob.glob(self.imgR_root)
        print("image_left: {}".format(images_left))
        images_left.sort()
        images_right.sort()
        print("共读取 {} 张左相机图像，{} 张右相机图像".format(len(images_left), len(images_right)))
        if len(images_left) == len(images_right):
            for index, _ in enumerate(images_left):
                print("index: {}".format(index))
                # print("\nRead image {}: \n{}\n{}".format(index, images_left[index], images_right[index]))
                img_l = cv2.imread(images_left[index])
                img_r = cv2.imread(images_right[index])
                img_l = cv2.resize(img_l, self.img_shape)
                img_r = cv2.resize(img_r, self.img_shape)

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

                # 查找棋盘格角点
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.corner_w, self.corner_h), None)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.corner_w, self.corner_h), None)

                # 如果找到棋盘格角点, 精细调整角点位置后, 添加对象点，图像点
                self.obj_points.append(self.objp)
                if ret_l is True:
                    if corners_l[0, 0, 0] < corners_l[-1, 0, 0]:
                        # print("*" * 5 + "order of {} is inverse!".format(i) + "*" * 5)
                        corners_l = np.flip(corners_l, axis=0).copy()
                    # 精细调整角点位置
                    cv2.cornerSubPix(gray_l, corners_l, winSize, (-1, -1), self.criteria)
                    self.img_points_l.append(corners_l)

                    # 绘制角点并显示
                    cv2.drawChessboardCorners(img_l, (self.corner_w, self.corner_h), corners_l, ret_l)
                    cv2.imshow("imgL", img_l)
                    cv2.waitKey(50)

                if ret_r is True:
                    if corners_r[0, 0, 0] < corners_r[-1, 0, 0]:
                        # print("*" * 5 + "order of {} is inverse!".format(i) + "*" * 5)
                        corners_r = np.flip(corners_r, axis=0).copy()
                    # 精细调整角点位置
                    cv2.cornerSubPix(gray_r, corners_r, winSize, (-1, -1), self.criteria)
                    self.img_points_r.append(corners_r)

                    # 绘制角点并显示
                    cv2.drawChessboardCorners(img_r, (self.corner_w, self.corner_h), corners_r, ret_r)
                    cv2.imshow("imgR", img_r)
                    cv2.waitKey(50)
            cv2.destroyAllWindows()
            # print("self.obj_points_len", len(self.obj_points))
            # print("self.img_points_r_len", len(self.img_points_r))
            # print("self.img_points_l_len", len(self.img_points_l))
        else:
            raise ValueError("左图像和右图像数量不一致!")

    def stereo_calibrate(self, img_shape, alpha=0):
        """
        执行双目相机标定。
        :param img_shape: 图像尺寸，格式为 (宽度, 高度)，用于标定和校正过程。
        :param alpha: 拉伸参数，控制校正后的图像范围。
        """
        print("开始双目相机标定")
        flag1 = 0
        flag1 |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        _, M1, self.D1, _, _ = cv2.calibrateCamera(self.obj_points,
                                                   self.img_points_l,
                                                   img_shape,
                                                   None,
                                                   None,
                                                   flags=flag1)
        _, M2, self.D2, _, _ = cv2.calibrateCamera(self.obj_points,
                                                   self.img_points_r,
                                                   img_shape,
                                                   None,
                                                   None,
                                                   flags=flag1)

        flag2 = 0
        flag2 |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
        ret, self.K1, self.D1, self.K2, self.D2, self.R, self.T, _, _ = cv2.stereoCalibrate(
            self.obj_points,
            self.img_points_l,
            self.img_points_r,
            M1,
            self.D1,
            M2,
            self.D2,
            img_shape,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
            flags=flag2)

        # 进行立体更正
        self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.K1, self.D1,
            self.K2, self.D2,
            self.img_shape,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha)

        # 输出结果
        print("左相机内参:\n", self.K1)
        print("左相机畸变系数:\n", self.D1)
        print("右相机内参:\n", self.K2)
        print("右相机畸变系数:\n", self.D2)
        print("旋转矩阵 R:\n", self.R)
        print("平移向量 T:\n", self.T)
        print("矫正变换矩阵 R1:\n", self.R1)
        print("重投影矩阵 P1:\n", self.P1)
        print("矫正变换矩阵 R2:\n", self.R2)
        print("重投影矩阵 P2:\n", self.P2)
        print("标定完成")

    def stereo_Rectify(self, img_l, img_r, shape=(720, 1280)):
        """
        对左右图像进行立体校正。
        :param img_l: 左图像。
        :param img_r: 右图像。
        :param shape: 输出图像尺寸。
        :return: 校正后的左右图像。
        """
        print("对左右图像进行立体校正")
        img_l = cv2.resize(img_l, (shape[1], shape[0]))
        imgL_rec = rectify_image(img_l, self.K1, self.D1, self.R1, self.P1)

        img_r = cv2.resize(img_r, (shape[1], shape[0]))
        imgR_rec = rectify_image(img_r, self.K2, self.D2, self.R2, self.P2)
        map_x = generate_coordinate_grid(shape=shape, axis=0)
        map_y = generate_coordinate_grid(shape=shape, axis=1)
        print(self.Calib)
        map_x = map_x - (float(self.Calib["u0l"]) - float(self.Calib["u0r"]))

        map_x = map_x.astype("float32")
        map_y = map_y.astype("float32")
        imgR_rec = cv2.remap(imgR_rec, map_x, map_y, cv2.INTER_LINEAR)

        return imgL_rec, imgR_rec

    def read_yaml_file(self, yaml_file):
        """
        从 YAML 文件中读取标定结果。
        :param yaml_file: 文件路径。
        """
        # 使用 ruamel.yaml 加载 YAML 文件
        yaml = YAML()
        with open(yaml_file, 'r') as file:
            data = yaml.load(file)

        # 提取数据
        camera_calibration = data.get("camera_calibration", {})
        print(camera_calibration)

        # Calib 部分
        self.Calib = camera_calibration.get("Calib", {})
        print(self.Calib)

        # Left Camera 部分
        left_camera = camera_calibration.get("left_camera", {})
        self.K1 = np.array(left_camera.get("k"))  # 转换为 NumPy 数组
        self.D1 = np.array(left_camera.get("d"))

        # Right Camera 部分
        right_camera = camera_calibration.get("right_camera", {})
        self.K2 = np.array(right_camera.get("k"))
        self.D2 = np.array(right_camera.get("d"))

        # Extrinsic Parameters 部分
        extrinsic_parameters = camera_calibration.get("extrinsic_parameters", {})
        self.R = np.array(extrinsic_parameters.get("R"))
        self.T = np.array(extrinsic_parameters.get("T"))

        # Rectification Parameters 部分
        rectification_parameters = camera_calibration.get("Rectification_Parameters", {})
        self.R1 = np.array(rectification_parameters.get("R1"))
        self.P1 = np.array(rectification_parameters.get("P1"))
        self.R2 = np.array(rectification_parameters.get("R2"))
        self.P2 = np.array(rectification_parameters.get("P2"))

        print("标定数据已成功加载！")

    def save_yaml_file(self, yaml_file):
        """
        将标定结果保存到 YAML 文件中。
        :param yaml_file: 保存的文件路径。
        """
        print("将标定结果保存到 YAML 文件中")
        u0l = float(self.P1[0, 2])  # 转换为 Python 原生 float
        u0r = float(self.P2[0, 2])
        v0 = float(self.P1[1, 2])
        baseline = float(1.0 / self.Q[3, 2])
        focus = float(self.P1[0, 0])
        self.Calib = {
            "u0l": u0l,
            "u0r": u0r,
            "v0": v0,
            "b_line": baseline,
            "focus": focus
        }
        data = {
            "camera_calibration": {
                "Calib": {
                    "u0l": u0l,
                    "u0r": u0r,
                    "v0": v0,
                    "b_line": baseline,
                    "focus": focus
                },
                "left_camera": {
                    "k": self.K1.tolist(),
                    "d": self.D1.tolist()
                },
                "right_camera": {
                    "k": self.K2.tolist(),
                    "d": self.D2.tolist()
                },
                "extrinsic_parameters": {
                    "R": self.R.tolist(),
                    "T": self.T.tolist()
                },
                "Rectification_Parameters": {
                    "R1": self.R1.tolist(),
                    "P1": self.P1.tolist(),
                    "R2": self.R2.tolist(),
                    "P2": self.P2.tolist()
                },
                "Disparity_Depth_Matrix":{
                    "Q": self.Q.tolist()
                }
            }
        }
        # 使用 ruamel.yaml 写入文件
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)  # 设置缩进格式
        # 将数据写入 YAML 文件
        with open(yaml_file, "w") as file:
            yaml.dump(data, file)
        print("标定数据已成功保存到文件！")

    '''立体标定'''
    def Run_Calibrator(self, alpha=0):
        """
        运行标定流程。
        :param alpha: 拉伸参数。
        """
        if os.path.exists(self.yaml_file):
            print("标定数据已存在，从文件中读取...")
            self.read_yaml_file(self.yaml_file)
        else:
            # calibration.
            print("标定文件不存在，开始标定...")
            self.read_images(winSize=(5, 5))
            self.stereo_calibrate(self.img_shape, alpha=alpha)
            self.save_yaml_file(self.yaml_file)


if __name__ == '__main__':
    # 参数设置
    img_H, img_W = 480, 640  # 图像分辨率
    img_data_path = "./data/"  # 数据根目录
    calibrator = StereoCameraDemarcate(
        data_root=img_data_path, # 数据根目录
        square_size=20, # 标定板方格尺寸（单位：毫米）
        corner_size=(8, 5),  # 标定板角点行列数
        suffix="jpg",  # 图像文件后缀
        image_shape=(img_H, img_W), # 图像尺寸
        camera_param_file="StereoParameters_03" # 标定参数文件
    )

    # 执行标定
    calibrator.Run_Calibrator(alpha=1)

    # 测试校正效果
    print("测试矫正效果···")
    images_r = glob.glob(os.path.join(img_data_path, 'left/*.jpg'))
    images_l = glob.glob(os.path.join(img_data_path, 'right/*.jpg'))
    images_l.sort()
    images_r.sort()
    imgL, imgR = cv2.imread(images_l[0]), cv2.imread(images_r[0])
    # 对左右图像进行立体校正
    imgL_Rectify, imgR_Rectify = calibrator.stereo_Rectify(imgL, imgR, shape=(img_H, img_W))
    # 显示校正后的图像
    combined_img = np.hstack((imgL_Rectify, imgR_Rectify))
    # 计算每条线之间的间距
    num_lines = 30
    spacing = 720 // num_lines
    # 绘制横线
    for i in range(num_lines):
        y = i * spacing
        cv2.line(combined_img, (0, y), (2560, y), (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('Rectified Images', combined_img)
    cv2.imwrite('./rectified.jpg', combined_img)
    # cv2.imshow("imgL_Rectify", imgL_Rectify)
    # cv2.imshow("imgR_Rectify", imgR_Rectify)
    cv2.waitKey(0)
