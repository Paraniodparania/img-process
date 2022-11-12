import cv2
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from scipy import signal
import pywt
import random

def division(img):
    row, col = img.shape
    begin_r = begin_c = 0
    end_r = end_c = 8
    t = []
    for i in range(int(row * col / 64)):
        if begin_r == row:
            begin_r = 0
            end_r = 8
            begin_c = end_c
            end_c += 8
        block = img[begin_c:end_c, begin_r:end_r]  # 二维数组切片操作 以逗号换维，第一维是行 第二维是列
        t.append(block)
        begin_r = end_r
        end_r += 8
    return t

def svd(M):
    """
    Args:
        M: numpy matrix of shape (m, n)
    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u, s, v = np.linalg.svd(M)

    return u, s, v

def con_temp(list_u,list_v):
    temp = np.zeros((32, 32), dtype='int')
    # 选择第一列第二行元素比较
    for locate in range(len(list_u)):
        u = list_u[locate][1][0]
        v = list_v[locate][1][0]
        i = int(locate / 32)  # 行号
        j = locate % 32  # 列号
        if u >= v:
            temp[i][j] = 1
        else:
            temp[i][j] = 0
    return temp

def con_S(temp, WaterMark):
    row, col = WaterMark.shape
    for i in range(row):
        for j in range(col):
            if WaterMark[i][j] == 255:
                WaterMark[i][j] = 1
    S = np.zeros((32, 32), dtype='int')
    for i in range(row):
        for j in range(col):
            t = XOR(temp[i][j], WaterMark[i][j])
            S[i][j] = t
    return S

def Arnold(img):
    row, col = img.shape
    p = np.zeros((row, col))
    a = 1
    b = 1
    for i in range(row):
        for j in range(col):
            x = (i + b * j) % row
            y = (a * i + (a * b + 1) * j) % col
            p[x, y] = img[i, j]
    return p

def dearnold(img):
    row, col = img.shape
    p = np.zeros((row, col))
    a = 1
    b = 1
    for i in range(row):
        for j in range(col):
            x = ((a * b + 1) * i - b * j) % row
            y = (-a * i + j) % col
            p[x, y] = img[i, j]
    return p

def AND(x1, x2):
    # 判断条件(x1w1+x2w2)>1 retuen 1,否则return 0
    x = np.array([x1, x2])
    w = np.array([0.6, 0.5])
    y = np.sum(x * w)
    if y > 1:
        return 1
    else:
        return 0

# numpy实现非运算
def NOT(x1, x2):
    # 判断条件(x1w1+x2w2)<1 retuen 1,否则return 0
    x = np.array([x1, x2])
    w = np.array([0.6, 0.5])
    y = np.sum(x * w)
    if y <= 1:
        return 1
    else:
        return 0

# numpy实现或运算
def OR(x1, x2):
    # 权重x1,x2要大于偏执b,判断条件 为真-> b+x1w1+x2w2>=0 为假->  b+x1w1+x2w2<=0
    x = np.array([x1, x2])
    w = np.array([0.2, 0.5])
    b = -0.1
    y = np.sum(x * w) + b
    if y <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    #判断条件将输入值x1,x2进行非not运算和或or运算然后再将其返回值进行与and运算变得到异或xor运算
    m=NOT(x1,x2)
    n=OR(x1,x2)
    k =AND(m,n)
    return k

def WaterMark_extract(Carrier,S):
    # 构造temp矩阵
    LL, (LH, HL, HH) = pywt.dwt2(Carrier, 'db1')  # LL-256 * 256
    # LL分块
    list = division(LL)
    # 对每个分块使用SVD分解
    list_u = []
    list_v = []
    for i in range(len(list)):
        u, s, v = svd(list[i])
        list_u.append(u)
        list_v.append(v)

    temp = np.zeros((32, 32), dtype='int')
    temp = con_temp(list_u, list_v)

    WaterMark = np.zeros((32, 32), dtype='int')
    WaterMark = con_S(temp, S)
    for i in range(20):
        WaterMark = dearnold(WaterMark)
    return WaterMark

def nc(WaterMark, WaterMark_ex):
    row, col = WaterMark.shape
    s = s1 = s2 = 0
    for i in range(col):
        for j in range(row):
            s1 += WaterMark[i][j] ** 2
            s2 += WaterMark_ex[i][j] ** 2
            s += WaterMark[i][j] * WaterMark_ex[i][j]
    s1 = s1 ** 0.5
    s2 = s2 ** 0.5
    nc = s / (s1 * s2)
    return nc

def gaussian_noise(image, mean=0, var=0.01):
    # 添加高斯噪声
    # mean : 均值
    # var : 方差
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = out * 255
    return out

def sp_noise(image, prob):
    # 添加椒盐噪声
    # prob:噪声比例
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def correl2d(img, window):
    mode = 'same'
    boundary = 'fill'
    s = signal.correlate2d(img, window, mode='same', boundary='fill')
    return s.astype(np.uint8)

def crop(img, a, b, x, y):
    '''
    :param a: 裁剪行起始位置
    :param b: 裁剪行终止位置
    :param x: 裁剪列起始位置
    :param y: 裁剪列终止位置
    :return: 裁剪后的图像
    '''
    row, col = img.shape
    crop_image = np.zeros((row, col))
    for i in range(col):
        for j in range(row):
            crop_image[i,j] = img[i,j]
    i = x
    j = a
    while i < y:
        while j < b:
            crop_image[i][j] = 255
            j += 1
        i += 1
        j = a
    return crop_image

Carrier = io.imread('D:\python实例\imgProcess\standard_test_images\lena_gray_512.tif')
WaterMark = io.imread('D:\python实例\imgProcess\standard_test_images\zimua.bmp')

WaterMark1 = WaterMark

LL, (LH, HL, HH) = pywt.dwt2(Carrier, 'db1') # LL-256 * 256
# LL分块
list = division(LL)
# 对每个分块使用SVD分解
list_u = []
list_v = []
for i in range(len(list)):
    u, s, v = svd(list[i])
    list_u.append(u)
    list_v.append(v)

temp = np.zeros((32, 32), dtype='int')
temp = con_temp(list_u, list_v)

# 将水印图像置乱
for i in range(20):
    WaterMark = Arnold(WaterMark)

S = np.zeros((32, 32), dtype='int')
S = con_S(temp, WaterMark)

# WaterMark_embed = WaterMark_extract(Carrier, S)
# plt.imshow(WaterMark_embed, cmap='gray')

# 高斯噪声
# img_gaussian = gaussian_noise(Carrier, 0, 0.01)
# WaterMark_embed = WaterMark_extract(img_gaussian, S)
# plt.imshow(WaterMark_embed, cmap='gray')
# print(nc(WaterMark1, WaterMark_embed))

# 椒盐噪声
# img_sp = sp_noise(Carrier, 0.01)
# WaterMark_embed = WaterMark_extract(img_sp, S)
# plt.imshow(WaterMark_embed, cmap='gray')
# print(nc(WaterMark1, WaterMark_embed))

# 旋转攻击
# img_rotate = transform.rotate(Carrier, 15)
# WaterMark_embed = WaterMark_extract(img_rotate, S)
# plt.imshow(WaterMark_embed, cmap='gray')
# print(nc(WaterMark1, WaterMark_embed))

# 平均滤波
# window = np.ones((5, 5)) / (5 ** 2)
# img_lv = correl2d(Carrier, window)
# WaterMark_embed = WaterMark_extract(img_lv, S)
# plt.imshow(WaterMark_embed, cmap='gray')
# print(nc(WaterMark1, WaterMark_embed))

# 剪切攻击
# img_crop = crop(Carrier, 0, 256, 0, 256)
# WaterMark_embed = WaterMark_extract(img_crop, S)
# plt.imshow(WaterMark_embed, cmap='gray')
# print(nc(WaterMark1, WaterMark_embed))
