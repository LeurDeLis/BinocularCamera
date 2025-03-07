# author: young
import cv2
import numpy as np
from ruamel.yaml import YAML

# 使用 ruamel.yaml 加载 YAML 文件
yaml = YAML()
yaml_file = '../CameraParam/StereoParameters_new.yaml'
with open(yaml_file, 'r') as file:
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

size = (640, 480)  # open windows size

# R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix,  # 左摄像机内参
                                                                  left_distortion,  # 左摄像机畸变系数
                                                                  right_camera_matrix,  # 右摄像机内参
                                                                  right_distortion,  # 右摄像机畸变系数
                                                                  size,  # 图像尺寸
                                                                  R,  # 旋转矩阵
                                                                  T  # 平移矩阵
                                                                  )

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,  # 左摄像机内参
                                                   left_distortion,  # 左摄像机畸变系数
                                                   R1,  # 左摄像机旋转矩阵
                                                   P1,  # 左摄像机投影矩阵
                                                   size,  # 图像尺寸
                                                   cv2.CV_16SC2  # 数据类型
                                                   )
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)
