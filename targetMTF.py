import cv2
import numpy as np
from scipy.signal import windows, convolve2d
import matplotlib.pyplot as plt
from SFR import SFR  # 导入 SFR 类

# 1. 手动选择ROI
def select_roi(image):
    roi_coords = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi_coords)
    roi = image[y:y + h, x:x + w]
    return roi

# 2. OECF校正
def oecf_correction(image):
    gamma = 1.5
    if len(image.shape) == 3:  # 彩色图像
        red = np.power(image[:, :, 2] / 255.0, gamma) * 255.0
        green = np.power(image[:, :, 1] / 255.0, gamma) * 255.0
        blue = np.power(image[:, :, 0] / 255.0, gamma) * 255.0
        corrected_image = np.stack([red, green, blue], axis=-1)
    else:  # 灰度图像
        corrected_image = np.power(image / 255.0, gamma) * 255.0
        corrected_image = np.stack([corrected_image, corrected_image, corrected_image], axis=-1)  # 转换为三维数组
    return corrected_image.astype(np.uint8)

# 11. 目标MTF
def target_mtf(f):
    return 1 - 0.1030396 * f + 0.00422688 * f ** 2 - 0.0000915521 * f ** 3 + 0.00000103079 * f ** 4

def calculate_average_sfr(esfr_g, esfr_y):
    # 计算 G 通道 SFR 的均值
    average_sfr_g = sum(esfr_g) / len(esfr_g)
    # 计算 Y 通道 SFR 特性的均值
    average_sfr_y = sum(esfr_y) / len(esfr_y)
    return average_sfr_g, average_sfr_y

# 12. 计算MTF
def calculate_system_mtf(esfr):
    # 使用测量的系统 SFR 特性
    return esfr

def calculate_scanner_mtf(system_mtf, target_mtf_values, direction='horizontal'):
        # 计算扫描仪 MTF 特性
        scanner_mtf = system_mtf / target_mtf_values
        return scanner_mtf

# 13. 应用MTF补偿
def apply_mtf_compensation(image, scanner_mtf, frequencies):
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)

    height, width = image.shape
    u = np.fft.fftshift(np.fft.fftfreq(width))
    v = np.fft.fftshift(np.fft.fftfreq(height))
    u, v = np.meshgrid(u, v)
    freq_magnitude = np.sqrt(u**2 + v**2)

    compensation_filter = np.interp(freq_magnitude.flatten(), frequencies, scanner_mtf, left=1, right=1)
    compensation_filter = compensation_filter.reshape(freq_magnitude.shape)

    # 避免除以零的情况
    compensated_fft = fft_shifted / (compensation_filter + 1e-10)  # 添加一个小值来避免除以零
    compensated_image = np.abs(np.fft.ifft2(np.fft.ifftshift(compensated_fft)))
    return compensated_image

# 主程序
if __name__ == "__main__":
    image_path = '../ES10000005_1200_Image2_1_MTF.tif'
    image = cv2.imread(image_path)

    # 选择 ROI
    roi = select_roi(image)

    # OECF 校正
    normalized_roi = oecf_correction(roi)

    # 创建 SFR 类的实例并设置参数
    sfr = SFR(oversampling=4, show_plots=0, difference_scheme='central', verbose=False, return_fig=False, quadratic_fit=True)

    # 计算各通道的 SFR
    mtf_r, _ = sfr.calc_sfr(normalized_roi[:, :, 2])  # R 通道
    mtf_g, _ = sfr.calc_sfr(normalized_roi[:, :, 1])  # G 通道
    mtf_b, _ = sfr.calc_sfr(normalized_roi[:, :, 0])  # B 通道

    # 计算亮度（Luminance）的 SFR
    luminance = 0.2126 * normalized_roi[:, :, 2] + 0.7152 * normalized_roi[:, :, 1] + 0.0722 * normalized_roi[:, :, 0]
    mtf_luminance, _ = sfr.calc_sfr(luminance)

    # 提取 MTF 和频率
    frequencies_cycles_per_pixel = mtf_r[:, 0]
    mtf_values_r = mtf_r[:, 1]
    mtf_values_g = mtf_g[:, 1]
    mtf_values_b = mtf_b[:, 1]
    mtf_values_luminance = mtf_luminance[:, 1]

    # 假设图像的像素密度为 300 DPI
    dpi = 300
    # 转换为每毫米像素数
    pixels_per_mm = dpi / 25.4
    # 转换频率单位
    frequencies_cycles_per_mm = frequencies_cycles_per_pixel * pixels_per_mm
    # 计算目标 MTF
    target_mtf_values = target_mtf(frequencies_cycles_per_mm)

    # 计算扫描仪 MTF
    system_mtf_g_horizontal = calculate_system_mtf(mtf_values_g)
    scanner_mtf_g_horizontal = calculate_scanner_mtf(system_mtf_g_horizontal, target_mtf_values, direction='horizontal')

    system_mtf_y_horizontal = calculate_system_mtf(mtf_values_luminance)
    scanner_mtf_y_horizontal = calculate_scanner_mtf(system_mtf_y_horizontal, target_mtf_values, direction='horizontal')

    # 计算垂直方向的 MTF
    system_mtf_g_vertical = calculate_system_mtf(mtf_values_g)
    scanner_mtf_g_vertical = calculate_scanner_mtf(system_mtf_g_vertical, target_mtf_values, direction='vertical')

    system_mtf_y_vertical = calculate_system_mtf(mtf_values_luminance)
    scanner_mtf_y_vertical = calculate_scanner_mtf(system_mtf_y_vertical, target_mtf_values, direction='vertical')

    # 绘制结果
    plt.figure()

    plt.plot(frequencies_cycles_per_mm, mtf_values_g, label='G Channel')
    plt.plot(frequencies_cycles_per_mm, mtf_values_luminance, label='Luminance')
    plt.plot(frequencies_cycles_per_mm, target_mtf_values, label='Target MTF')
    plt.plot(frequencies_cycles_per_mm, scanner_mtf_g_horizontal, label='Scanner MTF (G Channel, Horizontal)')
    plt.plot(frequencies_cycles_per_mm, scanner_mtf_y_horizontal, label='Scanner MTF (Luminance, Horizontal)')
    plt.plot(frequencies_cycles_per_mm, scanner_mtf_g_vertical, label='Scanner MTF (G Channel, Vertical)')
    plt.plot(frequencies_cycles_per_mm, scanner_mtf_y_vertical, label='Scanner MTF (Luminance, Vertical)')
    plt.xlabel('Spatial Frequency (cycles/mm)')
    plt.ylabel('Response')
    plt.title('SFR for R, G, B Channels and Luminance')
    plt.legend()
    plt.grid()

    # MTF 补偿
    compensated_image_horizontal = apply_mtf_compensation(luminance, scanner_mtf_g_horizontal, frequencies_cycles_per_mm)
    compensated_image_vertical = apply_mtf_compensation(luminance, scanner_mtf_g_vertical, frequencies_cycles_per_mm)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original ROI (Luminance)")
    plt.imshow(luminance, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("MTF Compensated ROI (Luminance, Horizontal)")
    plt.imshow(compensated_image_horizontal, cmap='gray')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original ROI (Luminance)")
    plt.imshow(luminance, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("MTF Compensated ROI (Luminance, Vertical)")
    plt.imshow(compensated_image_vertical, cmap='gray')
    plt.show()

    # 保存 MTF 数据到文件
    with open('../scanner_mtf.txt', 'w') as file:
        file.write("SF\tGH\tGV\tYH\tYV\n")  # 写入表头
        for i in range(len(frequencies_cycles_per_mm)):
            file.write(
                f"{frequencies_cycles_per_mm[i]:.6f}\t\t{scanner_mtf_g_horizontal[i]:.6f}\t{scanner_mtf_g_vertical[i]:.6f}\t{scanner_mtf_y_horizontal[i]:.6f}\t{scanner_mtf_y_vertical[i]:.6f}\n"
            )

    # 生成 Target MTF 文件
    with open('../target_MTF.txt', 'w') as file:
        file.write("SF\tG\tY\n")  # 写入表头
        for i in range(len(frequencies_cycles_per_mm)):
            file.write(
                f"{frequencies_cycles_per_mm[i]:.6f}\t\t{target_mtf_values[i]:.6f}\t{mtf_values_luminance[i]:.6f}\n"  # 添加额外的空格
            )
