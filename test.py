# # Canny 边缘检测算法
import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('D:/Py project/Database/train/ISR_25/image_0004.jpg', 0)
# # # 灰度
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # 高斯滤波 卷积 3 * 3
# img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
#
# # 直接用高斯滤波结果进行边缘检测 阈值 50 ~ 150
# edge = cv2.Canny(img_blur, 50, 160, apertureSize=3, L2gradient=True)
# plt.imshow(edge, cmap='cividis')
# plt.colorbar()
# plt.show()
# #

# # # 显示atten_mean中像素分布直方图
# import matplotlib.pyplot as plt
# import torch
# atten_mean_cpu = atten_mean.cpu()
# atten_mean_scaled = atten_mean_cpu * 10000
# atten_mean_np = atten_mean_scaled.numpy()
# plt.hist(atten_se_mean_np.ravel(), bins=256, range=(0, 100), density=True, color='gray', alpha=0.7)
# plt.title('Pixel Value Histogram (Scaled)')
# plt.xlabel('Pixel Value')
# plt.ylabel('Normalized Frequency')
# plt.show()


# # 将atten_mean 经过canny边缘检测
# atten_se_mean = (atten_se_mean * 10000).astype(np.uint8)
# img_gray = cv2.cvtColor(atten_se_mean, cv2.COLOR_BGR2RGB)
# # # 高斯滤波 卷积 3 * 3
# img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
# # 直接用高斯滤波结果进行边缘检测 阈值 50 ~ 150
# edge = cv2.Canny(img_blur, 15, 160, apertureSize=3, L2gradient=True)
# plt.imshow(edge, cmap='cividis')
# plt.colorbar()
# plt.show()