import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2

# 定义 OECF 拟合函数（对数型模型或幂函数模型）
def oecf_model(log_exposure, a, b, c):
    # 使用对数函数拟合，确保 log_exposure + b 大于零
    return a * np.log(np.maximum(log_exposure + b, 1e-10)) + c  # 1e-10 保证数值始终大于零

# sRGB 传递函数 (gamma 曲线)
def srgb_transfer_function(x):
    return np.where(x <= 0.04045, 12.92 * x, 1.055 * np.power(x, 1/2.4) - 0.055)

# 1. 加载扫描图像
def load_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        raise ValueError("图像模式必须为 RGB24。")
    return np.array(image)

# 自动检测灰阶块区域的函数
def detect_patches(image_data, threshold=128, min_size=50, max_size=200000):
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h  # 宽高比
        print(f"检测到的区域 - 坐标: ({x}, {y}, {x+w}, {y+h}), 面积: {area}, 宽高比: {aspect_ratio:.2f}")
        if min_size < area < max_size and 0.8 < aspect_ratio < 1.2:  # 限制宽高比
            patch_coordinates.append((x, y, x + w, y + h))

    patch_coordinates = sorted(patch_coordinates, key=lambda coord: (coord[1], coord[0]))
    return patch_coordinates

# 可视化检测到的区域
def visualize_patches(image_data, patch_coordinates):
    visual_image = image_data.copy()
    for x1, y1, x2, y2 in patch_coordinates:
        cv2.rectangle(visual_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detected Patches", visual_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. 提取灰阶块区域的平均亮度
def extract_patch_means(image_data, patch_coordinates):
    mean_values = []
    for coord in patch_coordinates:
        x1, y1, x2, y2 = coord
        patch = image_data[y1:y2, x1:x2]
        mean_rgb = np.mean(patch, axis=(0, 1))  # 计算 RGB 平均值
        mean_values.append(mean_rgb)
    return np.array(mean_values)

# 3. 计算对数曝光值
def calculate_log_exposures(luminance_values):
    return np.log10(luminance_values / luminance_values[0])  # 归一化计算

# 4. 拟合 OECF
def fit_oecf(log_exposures, mean_values):
    """
    对每个通道拟合 OECF
    :param log_exposures: 对数曝光值 (logE)
    :param mean_values: 每个灰阶块的 RGB 平均值
    :return: 每个通道的拟合参数
    """
    params = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        # 使用 log_exposures 替换错误的 log_exposure
        popt, _ = curve_fit(oecf_model, log_exposures, mean_values[:, i])
        params[channel] = popt
    return params

# 5. 可视化 OECF 结果
def plot_oecf_and_srgb(log_exposures, mean_values, params):
    # 绘制 OECF 和 sRGB 曲线
    plt.figure(figsize=(8, 6))

    # OECF 曲线
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.plot(log_exposures, mean_values[:, i], 'o', label=f'{channel} Data')
        plt.plot(log_exposures, oecf_model(log_exposures, *params[channel]), '-', label=f'{channel} Fit')

    # sRGB 传递函数
    x_values = np.linspace(0, 1, 1000)  # sRGB 输入值范围
    y_values = srgb_transfer_function(x_values)  # 计算 sRGB 传递函数值
    plt.plot(x_values, y_values, '--', label='sRGB Transfer Function', color='black')

    # 设置图形标签
    plt.xlabel("Log Exposure")
    plt.ylabel("Digital Output")
    plt.title("Opto-Electronic Conversion Function (OECF) & sRGB Transfer Function")
    plt.legend()
    plt.grid()
    plt.show()

# 6. 按照 Appendix 2 格式保存 OECF 数据
def save_oecf_to_file(mean_values, output_path="C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/OECFfile/oecf_results.txt"):
    """
    按照 24790Amd_conformance_en 文档的 Appendix 2 要求保存 OECF 文件
    """
    with open(output_path, "w") as f:
        for r, g, b in mean_values:
            f.write(f"{r:.6f}\t{g:.6f}\t{b:.6f}\n")
    print(f"OECF 数据已保存到 {output_path}")

# 主函数
def main():
    # 图像路径
    image_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000001_600_Image1.tif"

    # 加载图像
    image_data = load_image(image_path)

    # 自动检测灰阶块的坐标
    patch_coordinates = detect_patches(image_data, min_size=50, max_size=200000)
    print(f"检测到的灰阶块坐标: {patch_coordinates}")

    # 可视化检测到的区域
    if patch_coordinates:
        visualize_patches(image_data, patch_coordinates)
    else:
        raise ValueError("未检测到有效的灰阶块，请检查图像质量或调整检测参数。")

    # 测试卡标称亮度值 (根据实际测试卡数据)
    luminance_values = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])

    # 调整灰阶块数量与亮度值一致
    patch_coordinates = patch_coordinates[:len(luminance_values)]

    # 提取灰阶块平均值
    mean_values = extract_patch_means(image_data, patch_coordinates)

    # 检查 mean_values 是否为空
    if len(mean_values) == 0:
        raise ValueError("灰阶块平均值计算失败，请检查图像和灰阶块检测逻辑。")

    # 计算对数曝光值
    log_exposures = calculate_log_exposures(luminance_values)

    # 拟合 OECF
    params = fit_oecf(log_exposures, mean_values)

    # 可视化 OECF 和 sRGB 传递函数
    plot_oecf_and_srgb(log_exposures, mean_values, params)

    # 保存 OECF 文件
    save_oecf_to_file(mean_values)

# 执行主程序
if __name__ == "__main__":
    main()
