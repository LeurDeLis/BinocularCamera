import cv2
import numpy as np
import time
import random
import math
from ruamel.yaml import YAML

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------

# 使用 ruamel.yaml 加载 YAML 文件
yaml = YAML()
yaml_file = '../CameraParam/StereoParameters_01.yaml'
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
# R1 = np.array(param['Rectification_Parameters']['R1'])
# P1 = np.array(param['Rectification_Parameters']['P1'])
# R2 = np.array(param['Rectification_Parameters']['R2'])
# P2 = np.array(param['Rectification_Parameters']['P2'])
# Q = np.array(param['Disparity_Depth_Matrix']['Q'])

# 图像尺寸
size = (640, 480)

# R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)
# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_32FC1)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_32FC1)


# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def on_mouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        three_D = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", three_D[y][x][0], three_D[y][x][1], three_D[y][x][2], "mm")
        print("世界坐标xyz 是：", three_D[y][x][0] / 1000.0, three_D[y][x][1] / 1000.0, three_D[y][x][2] / 1000.0, "m")

        distance = math.sqrt(three_D[y][x][0] ** 2 + three_D[y][x][1] ** 2 + three_D[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")


def on_blockSize_change(value):
    global blockSize
    # 并且在5到25之间
    blockSize = max(5, min(25, value))


def on_num_change(value):
    global num
    # 确保 num 大于0
    num = max(1, value)

# 加载视频文件
# capture = cv2.VideoCapture("./car.avi")
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 读取视频
fps = 0.0
ret, frame = capture.read()

# 初始化 blockSize 和 num 的默认值
blockSize = 9  # 初始值为9，范围在5-21之间
num = 10  # 初始值为10，必须大于0

# 创建滑动条
cv2.createTrackbar("Block Size", WIN_NAME, blockSize, 25, on_blockSize_change)
cv2.createTrackbar("Num", WIN_NAME, num, 50, on_num_change)

while ret:
    # 开始计时
    t1 = time.time()
    # 是否读取到了帧，读取到了则为True
    ret, frame = capture.read()
    # frame = cv2.imread('../CollectDataDemarcate/point_test.jpg')
    # 切割为左右两张图片
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:480, 640:1280]
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # # 转换为opencv的BGR格式
    # imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    # imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=num * 16,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # stereo = cv2.StereoSGBM_create(minDisparity=1,
    #                                numDisparities=num * 16,
    #                                blockSize=blockSize
    #                                )
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    # 完成计时，计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("depth", dis_color)
    cv2.imshow("left", frame1)
    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面

    # 鼠标回调事件
    cv2.setMouseCallback("left", on_mouse_pick_points, threeD)
    # 若键盘按下q则退出播放
    if cv2.waitKey(1) & 0xff == 27:
        break

# 释放资源
capture.release()
# 关闭所有窗口
cv2.destroyAllWindows()
