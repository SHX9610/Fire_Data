"""
*Preliminary* pytorch implementation.
data process for voxelmorph

底原始数据进行处理

裁剪

归一化: (1)局部归一化
      （2）全局归一化

"""

import numpy as np

def LocalNormalized(img):   # (X-单张图min)/(单张max-单张min)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def GlobalNormalized(img,global_min,global_max):   # 计算数据集所有图的像素最大值和最小值
    global_min.append(np.min(img))
    global_max.append(np.max(img))
    g_min = min(global_min)
    g_max = max(global_max)
    return g_min,g_max


