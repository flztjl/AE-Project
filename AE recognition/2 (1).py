import cv2
import numpy as np

# 设置棋盘的尺寸
width, height = 2100, 2970  # A1纸的尺寸，单位为毫米
square_size = 210  # 棋盘格的尺寸，单位为毫米

# 创建一个全白色的图像
image = np.ones((height, width), dtype=np.uint8) * 255

# 用黑色填充棋盘格
for i in range(0, height, square_size * 2):
    for j in range(0, width, square_size * 2):
        image[i:i+square_size, j:j+square_size] = 0
        image[i+square_size:i+square_size*2, j+square_size:j+square_size*2] = 0

# 保存图像
cv2.imwrite('chessboard_A1.png', image)
