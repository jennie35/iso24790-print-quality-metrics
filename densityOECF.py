import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import RectangleSelector
# 设置字体支持特殊符号
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False
# 全局变量
rectangles = []  # 存储选择的矩形区域
density_values = []  # 存储密度值


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


# 1. 加载图像并显示
image = cv2.imread('../img20241124_19151597.tif')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 显示图像
fig, ax = plt.subplots()
ax.imshow(image_rgb)

# 创建矩形选择器
rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5)

# 启用选择器并等待用户选择区域
plt.title('手动选择灰阶块')
plt.axis('off')
plt.show()

# 2. 手动输入密度值并计算亮度 Y 和 RGB 值
n_blocks = len(rectangles)  # 获取选择的矩形区域数
Y_values = []  # 用于存储亮度 Y
R_values = []  # 用于存储红色通道的值
G_values = []  # 用于存储绿色通道的值
B_values = []  # 用于存储蓝色通道的值

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

# 3. 根据密度值计算反射率（这里反射率的计算可以基于其他物理公式调整）
# 确保密度值在合理范围内（0 到 1.95）
density_values = np.clip(density_values, 0, 1.95)

# 打印每个灰阶块的密度值和反射率
print('密度值（Density）：')
for i in range(n_blocks):
    print(f"第 {i + 1} 个灰阶块的密度值: {density_values[i]}")

# 4. 绘制 OECF 曲线（连接散点）
plt.figure(figsize=(8, 6))

# 绘制OECF曲线
plt.plot(density_values, Y_values, '-o', label='亮度Y', linewidth=2, markersize=6)

# 绘制 RGB 通道值的曲线
plt.plot(density_values, R_values, 'r-', label='红色通道', linewidth=2, markersize=6)
plt.plot(density_values, G_values, 'g-', label='绿色通道', linewidth=2, markersize=6)
plt.plot(density_values, B_values, 'b-', label='蓝色通道', linewidth=2, markersize=6)

# 添加标签和图例
plt.xlabel('密度')
plt.ylabel('灰度级')
plt.legend()
plt.title('OECF曲线')
plt.grid(True)
plt.show()
