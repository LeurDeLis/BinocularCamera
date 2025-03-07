# -*- coding: UTF-8 -*-
"""
@Project ：BinocularCamera
@File    ：PointCloudInfo.py
@IDE     ：PyCharm
@Author  ：MFK
@Date    ：2025/3/4 上午10:09
"""

import cv2
import numpy as np
import open3d as o3d
from ruamel.yaml import YAML

# 读取 YAML 文件并提取标定参数
yaml = YAML()
yaml_file = '../CameraParam/StereoParameters_new.yaml'
with open(yaml_file, 'r') as file:
    data = yaml.load(file)

# 提取数据
param = data.get("camera_calibration", {})
# 左镜头的内参矩阵
left_camera_matrix = np.array(param['left_camera']['k'])
right_camera_matrix = np.array(param['right_camera']['k'])
# 畸变系数 (K1、K2、K3为径向畸变，P1、P2为切向畸变)
left_distortion = np.array(param['left_camera']['d'])
right_distortion = np.array(param['right_camera']['d'])
# 旋转矩阵和平移向量
R = np.array(param['extrinsic_parameters']['R'])
T = np.array(param['extrinsic_parameters']['T'])
# 校正参数
R1 = np.array(param['Rectification_Parameters']['R1'])
P1 = np.array(param['Rectification_Parameters']['P1'])
R2 = np.array(param['Rectification_Parameters']['R2'])
P2 = np.array(param['Rectification_Parameters']['P2'])
Q = np.array(param['Disparity_Depth_Matrix']['Q'])

# 图像分辨率（假设从 YAML 中获取）
image_size = (640, 480)  # 替换为实际的图像分辨率

# 计算校正映射
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, image_size, cv2.CV_32FC1
)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, image_size, cv2.CV_32FC1
)

# 全局变量用于存储滑动条的值
num = 3
block_size = 2


# 滑动条回调函数
def on_num_disparities_change(value):
    global num
    num = value


def on_block_size_change(value):
    global block_size
    block_size = value * 2 + 5  # 必须是奇数且大于等于5


# 主函数
def main():
    global num_disparities, block_size

    # 读取单张图像并分割为左右图像
    img = cv2.imread('../CollectDataDemarcate/data/test/img1.jpg')
    if img is None:
        print("无法加载图像，请检查路径是否正确！")
        return

    # 分割左右图像
    left_img = img[0:480, 0:640]  # 高度480，宽度640
    right_img = img[0:480, 640:1280]

    # 转换为灰度图像
    # left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    # right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 创建窗口和滑动条
    cv2.namedWindow('Disparity Map')
    cv2.createTrackbar('Num Disparities', 'Disparity Map', num, 25, on_num_disparities_change)
    cv2.createTrackbar('Block Size', 'Disparity Map', block_size, 20, on_block_size_change)

    while True:
        # 校正图像
        left_rectified = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)

        # 创建 StereoSGBM 对象
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num * 16,  # 必须是16的倍数
            blockSize=block_size,
        )

        # 计算视差图
        disparity = stereo.compute(cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

        # 显示视差图
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8U)
        cv2.imshow('Disparity Map', disparity_normalized)

        # 按下 'q' 键退出
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

        # 按下空格键保存点云
        if key == 32:
            # 调试：打印视差图的最小值和最大值
            print(f"Disparity min: {disparity.min()}, max: {disparity.max()}")

            # 过滤无效点
            mask = disparity > (disparity.min() + 1)  # 增加一个小的偏移量
            colors = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)  # 使用校正后的彩色图像
            colors = colors[mask]
            points_3d = cv2.reprojectImageTo3D(disparity, Q)
            points_3d = points_3d[mask]

            # 调试：打印点云和颜色数据的形状
            print(f"Points shape: {points_3d.shape}, Colors shape: {colors.shape}")

            if points_3d.size == 0 or colors.size == 0:
                print("点云或颜色数据为空，请检查视差图和过滤条件！")
                continue

            # 调整点云坐标系方向
            # 翻转 Z 轴，使点云朝前
            points_3d[:, 2] = -points_3d[:, 2]

            # 如果需要，可以进一步调整其他轴
            # points_3d[:, 0] = -points_3d[:, 0]  # 翻转 X 轴
            points_3d[:, 1] = -points_3d[:, 1]  # 翻转 Y 轴

            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化颜色值到 [0, 1]

            # 可视化点云
            o3d.visualization.draw_geometries([pcd])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
