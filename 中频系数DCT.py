import cv2
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


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


def embed(list_zig, WaterMark, k1, k2):
    row, col = WaterMark.shape
    l = []  # 作为暂存的列表
    list_zag = []  # 作为返回值返回
    z = 0
    a, b = 50, 1
    for i in range(col):
        for j in range(row):
            l = list_zig[z]
            if WaterMark[i][j] == 255:  # 1111 1111
                for i in range(28, 36):
                    r = l[i] - int(l[i])  # list_zig[i] 存放的是一个列表
                    x = int(l[i]) % 10
                    y = int(int(l[i]) / 10)
                    x1 = b * x + a * k2[0][i]
                    l[i] = y * 10 + x1 + r
                list_zag.append(l)
            else:
                for i in range(28, 36):
                    r = l[i] - int(l[i])
                    x = int(l[i]) % 10
                    y = int(int(l[i]) / 10)
                    x1 = b * x + a * k1[0][i]
                    l[i] = y * 10 + x1 + r
                list_zag.append(l)
            z += 1
    return list_zag


# 讲矩阵按照Z字形排列
def zigzag(data):
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    list = np.zeros(num, )
    k = 0
    i = 0
    j = 0

    while i < row and j < col and k < num:
        list[k] = data.item(i, j)
        k = k + 1
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i - 1) in range(row) and (j + 1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i - 1) not in range(row) and (j + 1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i - 1) not in range(row) and (j + 1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i + 1) in range(row) and (j - 1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i + 1) not in range(row) and (j - 1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i + 1) not in range(row) and (j - 1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1

    return list


def dezigzag(list):
    block = np.zeros((8, 8), dtype='float')
    row = block.shape[0]
    col = block.shape[1]
    num = row * col
    k = 0
    i = 0
    j = 0

    while i < row and j < col and k < num:
        block[i, j] = list[k]
        k = k + 1
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i - 1) in range(row) and (j + 1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i - 1) not in range(row) and (j + 1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i - 1) not in range(row) and (j + 1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i + 1) in range(row) and (j - 1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i + 1) not in range(row) and (j - 1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i + 1) not in range(row) and (j - 1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1

    return block


def WaterMark_extract(List_zig, k1, k2):
    WaterMark = np.zeros((64, 64), dtype='float')
    k1 = k1[0][28:36]
    k2 = k2[0][28:36]
    l = []  # 暂存每一个zig串
    cov = np.zeros((3,3),dtype='float') # 存放每一个协方差矩阵
    r = c = 0
    for i in range(len(List_zig)):
        p = q = 0
        l = List_zig[i][28:36]
        mat = np.stack((l,k1,k2),axis=0)
        cov = np.cov(mat)
        p = cov[0][1]
        q = cov[0][2]
        if p > q:
            WaterMark[c][r] = 0
        else:
            WaterMark[c][r] = 255
        r += 1
        if r == WaterMark.shape[0]:
            r = 0
            c += 1
    return WaterMark


WaterMark = io.imread('D:\python实例\imgProcess\standard_test_images\jsnormal.png')  # 255 = 1 0 = 0
Carrier = io.imread('D:\python实例\imgProcess\standard_test_images\lena_gray_512.tif')
# plt.imshow(Carrier,cmap='gray')
# plt.figure()
Carrier = Carrier.astype('float')

# 嵌入过程
# 分块DCT变换
list = division(Carrier)
list_DCT = []
block = np.zeros((8, 8), dtype='float')
for i in range(len(list)):
    block = list[i]
    list_DCT.append(cv2.dct(block))

# 生成随机序列
k1 = np.random.randn(1, 64)  # 0
k2 = np.random.randn(1, 64)  # 255

# 按照zig-zag排序的DCT系数中频部 分连续的8个系数作为嵌入的位置
list_zig = []
list_zag = []
for i in range(len(list_DCT)):
    list_zig.append(zigzag(list_DCT[i]))
list_zag = embed(list_zig, WaterMark, k1, k2)  # 这里是对原list_zig进行修改的

# 把以上每一个1×64的zig-zag序列转 换为8×8的DCT系数子块后进行逆DCT变换，将得图像子 块合并
list_block = []
for i in range(len(list_zag)):
    list_block.append(dezigzag(list_zag[i]))
list_iDCT = []
for i in range(len(list_block)):
    list_iDCT.append(cv2.idct(list_block[i]))
Carrier_embed = np.zeros((512, 512), dtype='float')
Merge(Carrier_embed, list_iDCT)

plt.imshow(Carrier_embed,cmap='gray')

# 提取过程
List = division(Carrier_embed)
List_DCT = []
for i in range(len(List)):
    List_DCT.append(cv2.dct(List[i]))

# 将分块后的矩阵转成zig序列
List_zig = []
for i in range(len(List)):
    List_zig.append(zigzag(List_DCT[i]))
WaterMark_ex = np.zeros((64, 64), dtype='float')
WaterMark_ex = WaterMark_extract(List_zig, k1, k2)
plt.imshow(WaterMark_ex, cmap='gray')
