import numpy as np
from skimage import io, transform
import cv2 as cv
from matplotlib import pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio
import random,math

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

# 嵌入到了第一行
def embed(S_W, S): # 将32个奇异值嵌入到第一行
    '''
    :param S_W: 水印图像的奇异值数组
    :param S: 每个子块的最大奇异值组成的矩阵S
    :return: 矩阵S’
    '''
    q = 0.1
    for i in range(len(S_W)):
        j = i
        S[j][i] = S[j][i] + q * S_W[i]
    return S

def con_diag(diag, list_s):
    z = 0
    for j in range(len(list_s)):
        diag[z][j] = list_s[j]
        z += 1
    return diag

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

def WaterMark_extract(Carrier_embed,Carrier, U_W, V_W):
# 嵌入水印的载体图像进行离散小波变换
    LL_e, (LH_e, HL_e, HH_e) = pywt.dwt2(Carrier_embed, 'db1')
# 子带进行分块处理
    List_e = []
    List_e = division(LL_e)
# 奇异值分解，取出最大奇异值S2
    List_es = []    # 子块最大奇异值保存在这里
    for i in range(len(List_e)):
        u, s, v = svd(List_e[i])
        List_es.append(s[0])
# 对原载体图像处理
    LL, (LH, HL, HH) = pywt.dwt2(Carrier, 'db1')
    List = []
    List = division(LL)
    List_s = []
    for i in range(len(List)):
        u, s, v = svd(List[i])
        List_s.append(s[0])
    S = np.zeros((32, 32), dtype='float')
    z = 0
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            S[i][j] = List_s[z]
            z += 1     # 构造近似矩阵
# 水印提取
    Sw = [] # 存放所有的奇异值，其中有嵌入的
    SVD = [] # 存放嵌入的奇异值信息
    z = 0
    q = 0.1
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            Sw.append((List_es[z] - S[i][j]) / q)
            z += 1
    # Sw = Sw[0:32]
    i = j = 0
    locate = 0
    while locate < 1023:
        locate = i * 32 + j
        SVD.append(Sw[locate])
        i += 1
        j += 1
    diag = np.zeros((32, 32), dtype='float')
    diag = con_diag(diag, SVD)
    WaterMark = np.dot(np.dot(U_W, diag), V_W)
    for i in range(10):
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

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

# 图片预处理
Carrier = io.imread('D:\python实例\imgProcess\standard_test_images\lena_gray_512.tif')
WaterMark = io.imread('D:\python实例\imgProcess\standard_test_images\zimua.bmp')
# 水印图像缩放为32 * 32
# WaterMark = cv.resize(WaterMark, (32, 32))
# ret,WaterMark = cv.threshold(WaterMark, 254,255,cv.THRESH_BINARY)
# plt.imshow(WaterMark,cmap='gray')

# 嵌入过程
LL, (LH, HL, HH) = pywt.dwt2(Carrier, 'db1')
list = division(LL)

# 组成矩阵S
list_u = []
list_s = []
list_v = []
for i in range(len(list)):
    u, s, v = svd(list[i])
    list_u.append(u)
    list_s.append(s)
    list_v.append(v)

S = np.zeros((32, 32), dtype='float')
z = 0
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        S[i][j] = list_s[z][0]
        z += 1

# 对水印图像进行k次置乱后,对置乱后的水印进行奇异值分解
for i in range(10):
    WaterMark = Arnold(WaterMark)

U_W, S_W, V_W = svd(WaterMark)

# :将水印嵌入到载体图像中
S = embed(S_W, S)
# 逆奇异值分解得到含载体图像的子块
# 将嵌入后的矩阵奇异值给每一个list_s数组
z = 0
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        list_s[z][0] = S[i][j]
        z += 1
# 构造嵌入后的子块
diag = np.zeros((8, 8), dtype='float')
block = np.zeros((8, 8), dtype='float')
list_block = []
for i in range(len(list_s)):
    diag = con_diag(diag, list_s[i])
    block = np.dot(np.dot(list_u[i], diag), list_v[i])
    list_block.append(block)
# 合并每个子块
LL_embed = np.zeros((256, 256), dtype='float')
Merge(LL_embed, list_block)

Carrier_embed = np.zeros((512, 512), dtype='float')
Carrier_embed = pywt.idwt2((LL_embed, (LH, HL, HH)), 'db1')

# plt.imshow(Carrier_embed, cmap='gray')
# plt.figure()


# 原水印图像提取
# WaterMark_ex = np.zeros((32, 32), dtype='float')
# WaterMark_ex = WaterMark_extract(Carrier_embed, Carrier, U_W, V_W)
# plt.imshow(Carrier_embed, cmap='gray')
# plt.figure()
# plt.imshow(WaterMark_ex, cmap='gray')
# print(peak_signal_noise_ratio(Carrier, Carrier_embed))
# print(nc(WaterMark, WaterMark_ex))
# 攻击处理
# 高斯噪声
# img_gaussian = np.zeros((32, 32), dtype='float')
# img_gaussian = gaussian_noise(Carrier_embed, 0, 0.001)
# # plt.imshow(img_gaussian, cmap='gray')
# WaterMark_gaussian = WaterMark_extract(img_gaussian, Carrier, U_W, V_W)
# plt.imshow(WaterMark_gaussian, cmap='gray')
# print(nc(WaterMark,WaterMark_gaussian))

# 椒盐噪声
# img_sp = np.zeros((32, 32), dtype='float')
# img_sp = sp_noise(Carrier_embed, 0.02)
# WaterMark_sp = WaterMark_extract(img_sp, Carrier, U_W, V_W)
# plt.imshow(WaterMark_sp, cmap='gray')
# print(nc(WaterMark,WaterMark_sp))

# 旋转攻击
img_rotate = np.zeros((32, 32), dtype='float')
img_rotate = rotate(Carrier_embed, 340)
# theta = np.linspace(0.,180., max(img_rotate.shape),endpoint =False)
# img_rotate = transform.radon(img_rotate, theta=theta)
WaterMark_rotate = WaterMark_extract(img_rotate, Carrier, U_W, V_W)
plt.imshow(img_rotate, cmap='gray')
plt.figure()
plt.imshow(WaterMark_rotate, cmap='gray')
print(nc(WaterMark, WaterMark_rotate))

# 剪切攻击
# img_crop = np.zeros((32, 32), dtype='float')
# img_crop = crop(Carrier_embed, 256, 512, 256, 512)
# plt.imshow(img_crop,cmap='gray')
# plt.figure()
# WaterMark_crop = WaterMark_extract(img_crop, Carrier, U_W, V_W)
# plt.imshow(WaterMark_crop,cmap='gray')
# print(nc(WaterMark,WaterMark_crop))

