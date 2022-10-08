# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2022/10/7 10:43

import cv2
import math
import numpy as np

def run_vidar_rec(spike_planes, gamma=None):

    # def gamma_trans(img, gamma):  # gamma函数处理
    #     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    #     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    #     return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

    # for i in range(len(spike_planes)-1):
    #     print((spike_planes[i] == spike_planes[i+1]).all())
    # 汇总每个像素点的脉冲次数
    img = (sum(spike_planes).astype(np.float32) / len(spike_planes)) * 255.

    # 自动计算gamma值
    if gamma is None:
        mean = np.mean(img)
        gamma = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

    # gamma变换
    rec_img = np.power(img / 255., gamma)

    return rec_img

if __name__ == '__main__':
    img_gray = cv2.imread("./test.jpg", 0)  # 灰度图读取，用于计算gamma值
    print(img_gray)
    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)
    print("mean:", mean)
    print("gamma_val:", gamma_val)

    img1 = np.power(img_gray / 255., 1 / 1.5)
    img2 = np.power(img_gray / 255., 1.5)
    print("img1:", img1.max(), img1.min())
    print("img2:", img2.max(), img2.min())

    cv2.imshow("raw_img", img_gray)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)