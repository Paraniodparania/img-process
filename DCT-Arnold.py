import cv2
import numpy as np
from skimage import io, color, util, transform, filters
from tqdm import trange
import cv2 as cv
import random
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


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


# division函数由按行不动 列先动分块
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


def DCT_2D(x):
    '''
    Discrete space cosine transform
    x: Input matrix
    '''
    N1, N2 = x.shape
    X = np.zeros((N1, N2))
    n1, n2 = np.mgrid[0:N1, 0:N2]
    for w1 in range(N1):
        for w2 in range(N2):
            l1 = (2 / N1) ** 0.5 if w1 else (1 / N1) ** 0.5
            l2 = (2 / N2) ** 0.5 if w2 else (1 / N2) ** 0.5
            cos1 = np.cos(np.pi * w1 * (2 * n1 + 1) / (2 * N1))
            cos2 = np.cos(np.pi * w2 * (2 * n2 + 1) / (2 * N2))
            X[w1, w2] = l1 * l2 * np.sum(x * cos1 * cos2)
    return X


def iDCT2D(X, shift=True):
    '''
    Inverse discrete space cosine transform
    X: Input spectrum matrix
    '''
    N1, N2 = X.shape
    x = np.zeros((N1, N2))
    k1, k2 = np.mgrid[0:N1, 0:N2]
    l1 = np.ones((N1, N2)) * (2 / N1) ** 0.5
    l2 = np.ones((N1, N2)) * (2 / N2) ** 0.5
    l1[0] = (1 / N1) ** 0.5;
    l2[:, 0] = (1 / N2) ** 0.5
    for n1 in range(N1):
        for n2 in range(N2):
            cos1 = np.cos(np.pi * k1 * (2 * n1 + 1) / (2 * N1))
            cos2 = np.cos(np.pi * k2 * (2 * n2 + 1) / (2 * N2))
            x[n1, n2] = np.sum(l1 * l2 * X * cos1 * cos2)
    return x


# 乘性扰动算法
def Mul_Disturbance(DCT_list, Watermark):
    # DCT_list.astype(float) # 第34个矩阵本应减小却增大 //0 为该处的像素值
    list_DCT_Mul = []
    a = 0.01  # 论文为0.04
    row, col = Watermark.shape
    z = 0
    for i in range(col):
        for j in range(row):
            if Watermark[i][j] == 0:
                list_DCT_Mul.append(DCT_list[z] * (1 - a))
            else:
                list_DCT_Mul.append(DCT_list[z] * (1 + a))
            z += 1
    return list_DCT_Mul


def Merge(Carrier, iDCT_list):
    row, col = Carrier.shape
    begin_r = begin_c = 0
    end_r = end_c = 8
    for i in range(len(iDCT_list)):
        if begin_r == row:
            begin_r = 0
            end_r = 8
            begin_c = end_c
            end_c += 8
        Carrier[begin_c:end_c, begin_r:end_r] = iDCT_list[i]
        begin_r = end_r
        end_r += 8


# 水印提取是水印嵌入的逆过程
def WaterMark_extract(Carrier_R, Carrier_DCT):  # 传参时候传入list 和 Carrier_DCT
    list_extract = []
    WaterMark = np.zeros((32, 32))
    block1 = np.zeros((8, 8), dtype='float')
    block2 = np.zeros((8, 8), dtype='float')
    block_original = np.zeros((8, 8))
    block_extract = np.zeros((8, 8))
    # 先进行分块
    list_original = division(Carrier_R)
    list_extract = division(Carrier_DCT)
    # 分别进行DCT变换
    for i in trange(len(list_original)):
        # list_extract[i] = DCT_2D(list_extract[i])
        # list_original[i] = DCT_2D(list_original[i])
        block1 = list_original[i]
        block2 = list_extract[i]
        list_original[i] = cv.dct(block1)
        list_extract[i] = cv.dct(block2)
    # 通过比较原图像与嵌入后的数值大小来判断水印值
    j = k = 0
    for i in range(len(list_original)):
        block_original = list_original[i]
        block_extract = list_extract[i]
        if block_original[0][0] < block_extract[0][0]:
            WaterMark[j][k] = 1  # 这里应该是1
        else:
            WaterMark[j][k] = 0  # 这里应该是0
        k += 1
        if k == 32:
            k = 0
            j += 1
    # Arnold 逆变换
    for i in range(10):
        WaterMark = dearnold(WaterMark)
    return WaterMark


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


def light_change(image, a):
    '''

    :param img: 载体图像
    :param a: 亮度变化系数
    :return: 改变后的图像
    '''
    row, col = image.shape
    for j in range(row):
        for i in range(col):
            image[i, j] = image[i, j] * a
    return image

def sp_noise(image,prob):

    '''
    添加椒盐噪声
    prob:噪声比例
    '''

    output = np.zeros(image.shape,np.uint8)
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


Carrier = io.imread('d:/python实例/imgProcess/DCTimg/lena_color_256.tif')  # 256 256 3

# 分割RGB图像为三个通道
Carrier_R = Carrier[:, :, 0]
Carrier_G = Carrier[:, :, 1]
Carrier_B = Carrier[:, :, 2]

WaterMark = io.imread('D:/python实例/imgProcess/DCTimg/hide.jpg')  # 其变成32 32
WaterMark = cv2.resize(WaterMark, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
Carrier_R = Carrier_R.astype(dtype='float')
WaterMark = color.rgb2gray(WaterMark)  # 32 32
ret, WaterMark = cv.threshold(WaterMark, 0.5, 1, cv2.THRESH_BINARY)
# plt.imshow(WaterMark,cmap='gray')
# plt.figure()

# 分割为8*8的block矩阵
list = division(Carrier_R)
#
#
# 自己的离散余弦变换
list_DCT = []
block = np.zeros((8, 8), dtype='float')
for i in trange(len(list)):
    # list_DCT.append(DCT_2D(list[i]))
    block = list[i]
    list_DCT.append(cv2.dct(block))

# Arnold变换
for i in range(10):
    WaterMark = Arnold(WaterMark)

# 乘性扰动
list_DCT_Mul = []
list_DCT_Mul = Mul_Disturbance(list_DCT, WaterMark)

# 离散余弦反变换
list_iDCT = []
for i in trange(len(list_DCT_Mul)):
    # list_iDCT.append(iDCT2D(list_DCT_Mul[i]))  # list_iDCT放置的是经过乘性扰动后的反余弦变换矩阵 大小8*8
    block = list_DCT_Mul[i]
    list_iDCT.append(cv.idct(block))

row, col = Carrier_R.shape
Carrier_DCT = np.zeros((row, col), dtype='float32')
Merge(Carrier_DCT, list_iDCT)  # 当前Carrier_DCT中存放着嵌入水印后的矩阵

# Carrier_DCT = Carrier_DCT.astype('')
# 添加噪声

# 添加高斯噪声
# mean = 0
# sigma = 6
# gauss = np.random.normal(mean, sigma, (row, col))
# image_gaussian = Carrier_DCT + gauss
# image_gaussian = np.clip(image_gaussian, a_min=0, a_max=255)
# plt.imshow(image_gaussian, cmap='gray')

# 添加椒盐噪声
# image_sp = sp_noise(Carrier_DCT,0.001)
# image_sp = image_sp.astype('float')
# plt.imshow(image_sp,cmap='gray')

# 旋转处理(30°)
# image_spin = transform.rotate(Carrier_DCT, 5)
# plt.imshow(image_spin,cmap='gray')
# #
# # 裁剪处理
image_crop = crop(Carrier_DCT,70,150,70,150)
# plt.imshow(image_crop,cmap='gray')
#
# 裁边攻击
#
# 锐化处理，采用拉普拉斯算子
# img_laplace = filters.laplace(Carrier_DCT,ksize=3  ,mask=None)
# img_enhance = Carrier_DCT + img_laplace
# plt.imshow(img_enhance,cmap='gray')
#
#
#
# plt.imshow(Carrier_DCT,cmap='gray')
# plt.figure()
#
# 提取水印图像
WaterMark = WaterMark_extract(Carrier_R, Carrier_DCT)
# plt.imshow(WaterMark, cmap='gray')

# 高斯噪声水印提取
# WaterMark_gaussian = WaterMark_extract(Carrier_R,image_gaussian)
# print(ssim(WaterMark,WaterMark_gaussian))
# plt.imshow(WaterMark_gaussian,cmap='gray')

#
# # 椒盐噪声水印提取
# WaterMark_sp = WaterMark_extract(Carrier_R,image_sp)
# print(ssim(WaterMark,WaterMark_sp))
# plt.imshow(WaterMark_sp,cmap='gray')
# plt.figure()
# #
# # 泊松噪声
# WaterMark_poisson = WaterMark_extract(Carrier_R,image_sp)
# plt.imshow(WaterMark_poisson,cmap='gray')
# plt.figure()
#
# # 旋转操作
# WaterMark_spin = WaterMark_extract(Carrier_R,image_spin)
# plt.imshow(WaterMark_spin,cmap='gray')

#
# # 裁剪操作
WaterMark_crop =WaterMark_extract(Carrier_R,image_crop)
print(ssim(WaterMark,WaterMark_crop))
plt.imshow(WaterMark_crop,cmap='gray')
# plt.figure()
#
# # 锐化处理
# WaterMark_laplace = WaterMark_extract(Carrier_R,img_laplace)
# plt.imshow(WaterMark_laplace,cmap='gray')
# plt.figure()
