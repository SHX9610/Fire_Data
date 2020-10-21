'''

可视化配准结果 ： 相加 看颜色 看神经血管的配准

'''

import numpy as np
import cv2
def addimage(img_1 ,img_2 ,img_3,k):     # img1:固定图像     img2：浮动图像    img3：配准后图像

    img_1 = np.squeeze(img_1) * 255
    img_2 = np.squeeze(img_2) * 255
    img_3 = np.squeeze(img_3) * 255
    # 将灰度图转换为彩图  底用蓝色，顶用红色  BGR
    img = np.zeros((2912,2912 ,3))
    img[: ,: ,0 ] =img_1.cpu().detach().numpy()  # 绿色
    img1 =img
    img = np.zeros((2912 ,2912 ,3))
    img[: ,: ,2 ] =img_2.cpu().detach().numpy()
    img2 =img  # 红色的浮动图像
    # 固定和浮动
    overlap1 =cv2.addWeighted(img1 ,0.5 ,img2 ,0.5 ,0)
    # 固定和配准后
    img[: ,: ,2 ] =img_3.cpu().detach().numpy()
    img3 = img
    overlap2 =cv2.addWeighted(img1 ,0.5 ,img3 ,0.5 ,0)
    # 浮动和配准后
    img = np.zeros((2912 ,2912 ,3))
    img[: ,: ,0 ] =img_2.cpu().detach().numpy()
    img2 =img
    img = np.zeros((2912 ,2912 ,3))
    img[: ,: ,2 ] =img_3.cpu().detach().numpy()
    img3 =img
    overlap3 =cv2.addWeighted(img2 ,0.5 ,img3 ,0.5 ,0)
    cv2.imwrite('D:\peizhunsd\data\\fixed_moved\\'+str(k)+'.png',overlap1)  # 固定+浮动
    cv2.imwrite('D:\peizhunsd\data\\fixed_registed\\'+str(k)+'.png',overlap2)  # 固定+配准后
    cv2.imwrite('D:\peizhunsd\data\moved_registed\\'+str(k)+'.png',overlap3)  # 浮动+配准后