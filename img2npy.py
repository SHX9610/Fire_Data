import numpy as np
import os
import cv2

from dataprocess import LocalNormalized
path = 'D:\peizhunsd\data\Images\\'
names = os.listdir(path)
for name in names:
    img_path = path + name
    img = cv2.imread(img_path,0)
    img = LocalNormalized(img)

    if name.split('.')[0].split('_')[-1] == '2':
        mov_path = 'D:\peizhunsd\data\mov\\'
        np.save(mov_path+name.split('.')[0]+'.npy',img)
    else:
        ref_path = 'D:\peizhunsd\data\\ref\\'
        np.save(ref_path + name.split('.')[0] + '.npy', img)


'''
# 使用全局归一化进行图像的预处理
from dataprocess import GlobalNormalized
path = 'D:\peizhunsd\data\Images\\'
names = os.listdir(path)

global_min = []
global_max = []
for name in names:  # 首先计算出整个数据集 像素最大值和最小值
    img_path = path + name
    img = cv2.imread(img_path,0)
    g_min,g_max = GlobalNormalized(img,global_min,global_max)

for name in names:
    img_path = path + name
    img = cv2.imread(img_path,0)
    img = (img - g_min) / (g_max - g_min)
    
    if name.split('.')[0].split('_')[-1] == '2':
        mov_path = 'D:\peizhunsd\data\mov\\'
        np.save(mov_path+name.split('.')[0]+'.npy',img)
    else:
        ref_path = 'D:\peizhunsd\data\\ref\\'
        np.save(ref_path + name.split('.')[0] + '.npy', img)
'''


