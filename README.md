

# 一、双目相机的标定（Python-OpenCV）

### 标定程序的使用

拿到源码之后，配置好Python环境，然后打开`CameraDemarcate.py`文件，对以下部分进行修改

```Python
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
```

需要根据我们的实际情况去修改数据集存放目录、标定使用的棋盘格方格尺寸、标定板的格点行列数（棋盘格行列数减一）、数据集存放格式（jpg、png等）、图像尺寸（640*480、1280\*720等），以及保存参数的文件名。

设置好之后直接运行即可。

运行：

```
 cd CollectDataDemarcate
 python3 CameraDemarcate.py
```

# 二、双目相机测距

## （一）算法应用

### 1. BM（Block Matching）算法实现测距

#### BM算法参数配置

在获取双目相机的校正参数之后，接下来需要进行立体匹配以计算视差图，从而进一步获取三维信息。为此，我们使用了OpenCV库中的`cv2.StereoBM_create`函数来创建一个BM（Block Matching）算法对象，用于立体匹配。

BM算法是一种基于块的匹配算法，它通过在左右图像中寻找匹配块来计算视差。以下是配置BM算法参数的详细步骤：

首先，我们计算视差范围`numberOfDisparities`，确保它是16的倍数，这是因为BM算法要求视差值必须是16的倍数。通过将图像宽度除以8并加上15，然后对结果取16的倍数，我们得到一个合理的视差范围。

接下来，我们创建一个`stereo`对象，并设置其参数。以下是各个参数的含义：

- `numDisparities`: 视差范围，表示最大视差值与最小视差值之差，乘以16是为了保证亚像素精度。
- `blockSize`: 匹配块的大小，通常取奇数，表示在左右图像中比较的块的大小。

然后，我们设置左右图像的有效像素区域、预处理滤波器的截断值、最小视差值、视差范围、纹理阈值、唯一性比率、斑点窗口大小和范围以及左右视差图的最大差异等参数，这些参数共同决定了视差图的计算质量。

具体使用方法如下：

```Python
# BM算法参数配置

# 计算视差范围，确保是16的倍数，1280是图像的宽度
numberOfDisparities = ((1280 // 8) + 15) & -16  # 640对应是分辨率的宽的一半，这里除以8是为了减少计算量，然后加15并取16的倍数是为了确保视差范围足够

# 创建StereoBM算法对象，用于立体匹配
stereo = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=9)  
# 设置左图像的有效像素区域
stereo.setROI1(camera_configs.validPixROI1)
# 设置右图像的有效像素区域
stereo.setROI2(camera_configs.validPixROI2)
# 设置预处理滤波器的截断值
stereo.setPreFilterCap(31)
# 设置匹配块的大小
stereo.setBlockSize(18)
# 设置最小视差值
stereo.setMinDisparity(0)
# 设置视差范围
stereo.setNumDisparities(numberOfDisparities)
# 设置纹理阈值，用于过滤纹理较少的区域
stereo.setTextureThreshold(10)
# 设置唯一性比率，用于过滤视差图中的噪声
stereo.setUniquenessRatio(15)
# 设置斑点窗口大小，用于过滤斑点噪声
stereo.setSpeckleWindowSize(100)
# 设置斑点范围，用于过滤斑点噪声
stereo.setSpeckleRange(32)
# 设置左右视差图的最大差异
stereo.setDisp12MaxDiff(1)
```

运行：

```
 cd BM
 python3 BM.py
```

### 2. SGBM（Semi-Global Block Matching）算法实现测距

SGBM算法适用于对深度精度要求较高、场景包含大量弱纹理区域或遮挡区域的情况。

#### SGBM算法参数配置

在获取双目相机的校正参数之后，接下来需要进行立体匹配以计算视差图，从而进一步获取三维信息。为此，我们使用了 OpenCV 库中的 `cv2.StereoSGBM_create` 函数来创建一个 SGBM（Semi-Global Block Matching）算法对象，用于立体匹配。SGBM 算法是一种基于块匹配并结合半全局优化的立体匹配算法，通过在多个方向上累积匹配代价来提高视差图的质量，尤其在弱纹理区域和遮挡区域表现优异。

具体使用方法如下：

```Python
img_channels = 3
stereo = cv2.StereoSGBM_create(minDisparity=1,			# 最小视差值
                               numDisparities=num * 16, # 视差范围，即最大视差值与最小视差值之差，必须是16的倍数
                               blockSize=blockSize, 	# 每个像素块的大小，必须是奇数
                               P1=8 * img_channels * blockSize * blockSize, 	# 视差平滑度约束项1
                               P2=32 * img_channels * blockSize * blockSize,    # 视差平滑度约束项2
                               disp12MaxDiff=-1, 				# 左右视差图的最大容许差异
                               preFilterCap=1, 					# 预滤波器窗口大小
                               uniquenessRatio=10,  			# 视差唯一性百分比
                               speckleWindowSize=100, 			# 视差连通区域大小
                               speckleRange=100,    			# 视差变化范围
                               mode=cv2.STEREO_SGBM_MODE_HH)    # 模式选择
```

运行：

```
 cd SGBM
 python3 SGBM.py
```

# 三、双目相机显示点云图

既然我们已经可以获取到图像的深度信息，那么就可以根据这些深度信息生成3D的点云图像。

这里引入一个新的python库`Open3D`库，`Open3D `是一个开源的 Python 和 C++ 库，专门用于 3D 数据处理，适用于点云、网格、体素等多种三维数据结构。它提供了高效的 3D 计算功能，如点云处理（降采样、去噪、分割、对齐）、表面重建、几何计算、可视化和机器学习集成。`Open3D `采用直观的 `Python API`，同时支持 `GPU `加速，使其在机器人、计算机视觉、3D 扫描和增强现实等领域广泛应用。此外，`Open3D `还能与深度学习框架（如 `PyTorch`、`TensorFlow`）集成，支持神经网络训练 3D 任务，提供 3D 形状识别、点云补全等高级功能。

该库具备灵活的可视化工具，可用于交互式查看 3D 物体，并支持与 `ROS `结合，实现实时 3D 传感数据处理。其模块化设计使得 `Open3D` 在 SLAM（同步定位与建图）、点云注册和 3D 物体识别等领域表现出色，成为 3D 计算研究和工业应用中的重要工具。

那我们要如何使用这个库去生成可视化的3D点云图呢？

生成点云图的步骤如下:

1）获取相机参数

2）获取立体校正所需的投影变换矩阵

3）获取去畸变和校正映射表

4）重建图像

5）SGBM算法参数配置

6）计算并显示视差图

7）获取点云的3D坐标

程序运行：

```bash
cd PointsCloudImg
python3 PointCloudInfo.py
```

# 四、YOLO目标检测+测距

## 1）YOLO的训练

参考我的另一个Github仓库[一个简单的YOLO训练](https://github.com/LeurDeLis/Simple-use-example-of-YOLOV8.git)

## 2）YOLO目标检测+双目测距实现

#### 1. 概述

这一小节详细说明如何使用 YOLOv8 目标检测算法结合双相模测距实现对目标的位置和测距。程序通过双相模课栈获取深度信息，将测距结果与 YOLOv8 检测结果进行融合，实现物体距离估计。

------

#### 2. 主体架构

系统包含以下三个主要模块：

1. **DistanceMeasurement** (双相模测距)：读取相机标定参数，进行异步校正和深度计算。
2. **YOLOV8_NCNN** (目标检测)：通过 YOLOv8 模型对物体进行检测，定位目标系统中心坐标。
3. **main** (主程序)：合并上述两个模块，完成实时检测与测距。

运行：

```bash
cd YOLO
python3 main.py
```
