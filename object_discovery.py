
import numpy as np
import cv2
import matplotlib.pyplot as plt


# canny边缘检测

def canny_edge(img, patch_size, atten_mean, ksize, thres1, thres2):
    scaled_atten_mean = (atten_mean.cpu().numpy() * 10000).astype(np.uint8)
    img_gray = cv2.cvtColor(scaled_atten_mean, cv2.COLOR_BGR2RGB)
    # # 高斯滤波 卷积 3 * 3
    img_blur = cv2.GaussianBlur(img_gray, ksize, 0)
    # 直接用高斯滤波结果进行边缘检测
    edge = cv2.Canny(img_blur, thres1, thres2, apertureSize=3, L2gradient=True)

    # 显示特征图的边缘检测结果
    # plt.imshow(edge, cmap='cividis')
    # plt.axis('off')
    # # plt.colorbar()
    # plt.show()
    # #

    object_num, labels, stats, centroids = cv2.connectedComponentsWithStats(edge, connectivity=8)
    return object_num, labels, stats, centroids
