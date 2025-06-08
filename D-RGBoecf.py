import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import RectangleSelector
import csv
from numpy.polynomial.polynomial import Polynomial

# 设置字体支持特殊符号
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False  # 设置负号正常显示

# 全局变量
rectangles = []  # 存储选择的矩形区域
density_values = []  # 存储密度值
Y_values = []  # 存储亮度 Y
R_values = []  # 存储 R 通道值
G_values = []  # 存储 G 通道值
B_values = []  # 存储 B 通道值
R_reflectance = []  # 红色通道反射率
G_reflectance = []  # 绿色通道反射率
B_reflectance = []  # 蓝色通道反射率

# 选择矩形区域的回调函数
def onselect(eclick, erelease):
    # 获取矩形区域的坐标 (x1, y1) 和 (x2, y2)
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    rectangles.append([x1, y1, abs(x2 - x1), abs(y2 - y1)])  # 存储选择的矩形区域
    # 在选择区域显示矩形框
    ax.add_patch(plt.Rectangle((x1, y1), abs(x2 - x1), abs(y2 - y1), edgecolor='r', facecolor='none', linewidth=2))
    plt.draw()

# 手动输入密度值的函数
def get_density_value(block_index):
    density = float(input(f"请输入第 {block_index + 1} 个灰阶块的密度值（0 到 1.95）："))
    return np.clip(density, 0, 1.95)  # 限制密度值范围

# 加权最小二乘法拟合
def weighted_least_squares(x, y, weights, degree=5):
    """
    进行加权最小二乘法回归拟合
    :param x: 输入数据
    :param y: 输出数据
    :param weights: 加权
    :param degree: 多项式的阶数
    :return: 拟合的多项式参数
    """
    return np.polyfit(x, y, degree, w=weights)

# 1. 加载图像并显示
image = cv2.imread('../img20241124_19151597.tif')  # 读取图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 显示图像
fig, ax = plt.subplots()
ax.imshow(image_rgb)

# 创建矩形选择器
rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5)

# 启用选择器并等待用户选择区域
plt.title('手动选择灰阶块')
plt.axis('off')  # 关闭坐标轴
plt.show()

# 2. 手动输入密度值并计算亮度 Y、RGB 值和反射率
n_blocks = len(rectangles)  # 获取选择的矩形区域数
for i in range(n_blocks):
    # 获取每个区域的 BoundingBox
    x, y, width, height = rectangles[i]

    # 提取区域并计算 RGB 三个通道的平均值
    region = image[y:y + height, x:x + width]  # 提取矩形区域
    avg_rgb = np.mean(region, axis=(0, 1))  # 在行列方向上计算均值

    # 提取 RGB 的通道值
    R, G, B = avg_rgb

    # 使用公式计算亮度 Y
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # 将计算的 Y、R、G、B 值存储
    Y_values.append(Y)
    R_values.append(R)
    G_values.append(G)
    B_values.append(B)

    # 获取并存储密度值
    density = get_density_value(i)
    density_values.append(density)

    # 将密度值转换为反射率，使用公式 R = 10^(-D)
    R_ref = 10 ** (-density) * R / 255  # 红色通道反射率
    G_ref = 10 ** (-density) * G / 255  # 绿色通道反射率
    B_ref = 10 ** (-density) * B / 255  # 蓝色通道反射率

    # 对反射率进行边界处理
    R_ref = np.clip(R_ref, 0.001, 0.933)
    G_ref = np.clip(G_ref, 0.001, 0.933)
    B_ref = np.clip(B_ref, 0.001, 0.933)

    # 存储调整后的反射率值
    R_reflectance.append(R_ref)
    G_reflectance.append(G_ref)
    B_reflectance.append(B_ref)

# 3. 打印反射率
print('反射率（Reflectance）：')
for i in range(n_blocks):
    print(
        f"第 {i + 1} 个灰阶块的红色反射率: {R_reflectance[i]:.4f}, 绿色反射率: {G_reflectance[i]:.4f}, 蓝色反射率: {B_reflectance[i]:.4f}")

# 4. 加权最小二乘法回归拟合
# 权重是反射率的倒数
R_weights = 1 / np.array(R_reflectance)
G_weights = 1 / np.array(G_reflectance)
B_weights = 1 / np.array(B_reflectance)

# 进行加权最小二乘法回归拟合
R_coeff = weighted_least_squares(R_values, R_reflectance, R_weights)
G_coeff = weighted_least_squares(G_values, G_reflectance, G_weights)
B_coeff = weighted_least_squares(B_values, B_reflectance, B_weights)

# 打印拟合的系数
print("\n加权最小二乘法回归拟合的结果：")
print(f"红色通道拟合的系数: {R_coeff}")
print(f"绿色通道拟合的系数: {G_coeff}")
print(f"蓝色通道拟合的系数: {B_coeff}")

# 5. 绘制 OECF 曲线（连接散点）
plt.figure(figsize=(8, 6))

# 绘制 OECF 曲线
plt.plot(density_values, Y_values, '-o', label='亮度Y', linewidth=2, markersize=6)

# 绘制 RGB 通道值的曲线
plt.plot(density_values, R_values, 'r-', label='红色通道', linewidth=2, markersize=6)
plt.plot(density_values, G_values, 'g-', label='绿色通道', linewidth=2, markersize=6)
plt.plot(density_values, B_values, 'b-', label='蓝色通道', linewidth=2, markersize=6)

# 添加标签和图例
plt.xlabel('密度')
plt.ylabel('亮度Y')
plt.legend()
plt.title('OECF曲线')
plt.grid(True)
plt.show()

# 6. 绘制反射率与 RGB 值的关系图（加权最小二乘法拟合）
plt.figure(figsize=(8, 6))

# 绘制加权拟合结果
plt.plot(R_values, R_reflectance, 'ro', label='红色通道反射率 (原始)', markersize=6)
plt.plot(G_values, G_reflectance, 'go', label='绿色通道反射率 (原始)', markersize=6)
plt.plot(B_values, B_reflectance, 'bo', label='蓝色通道反射率 (原始)', markersize=6)

# 绘制加权最小二乘法回归的拟合曲线
x_range = np.linspace(min(R_values), max(R_values), 100)
plt.plot(x_range, np.polyval(R_coeff, x_range), 'r-', label='红色通道反射率 (拟合)', linewidth=2)
plt.plot(x_range, np.polyval(G_coeff, x_range), 'g-', label='绿色通道反射率 (拟合)', linewidth=2)
plt.plot(x_range, np.polyval(B_coeff, x_range), 'b-', label='蓝色通道反射率 (拟合)', linewidth=2)

# 添加标签和图例
plt.xlabel('RGB 值')
plt.ylabel('反射率')
plt.title('RGB 值与反射率的关系（加权最小二乘法拟合）')
plt.legend()
plt.grid(True)
plt.show()

# 7. 保存反射率值到 CSV 文件
with open('../reflectance_values.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['R Reflectance', 'G Reflectance', 'B Reflectance'])  # 不再包含 Block Index

    for i in range(n_blocks):
        writer.writerow([R_reflectance[i], G_reflectance[i], B_reflectance[i]])

print("反射率已成功保存到 'reflectance_values.csv' 文件中。")
