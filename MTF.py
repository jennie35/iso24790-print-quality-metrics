import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.fft import fft, fftshift, fftfreq, ifft2


# 获取垂直斜边的ROI
def get_roi(image, x_start, y_start, width, height):
    # 截取图像中的ROI区域
    roi = image[y_start:y_start + height, x_start:x_start + width]
    if roi.ndim == 2:  # 如果是灰度图像，将其转为RGB格式
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    return roi


# 计算图像每一行的像素矩心
def compute_centroids(image):
    # 计算每行像素的矩心
    centroids = []
    for i in range(image.shape[0]):
        row = image[i, :]
        centroid = np.sum(np.arange(len(row)) * row) / np.sum(row)  # 计算矩心
        centroids.append(centroid)
    return np.array(centroids)


# 对每行的矩心进行线性拟合
def fit_line(centroids):
    # 使用最小二乘法对矩心进行线性拟合
    x = np.arange(len(centroids))
    coeffs = np.polyfit(x, centroids, 1)
    return coeffs


# 重新定位ROI并获得ESF
def compute_esf(roi):
    # 计算ROI的边缘扩展函数（ESF）
    centroids = compute_centroids(roi)
    coeffs = fit_line(centroids)
    esf = centroids - np.polyval(coeffs, np.arange(len(centroids)))  # ESF = 矩心 - 拟合直线
    return esf


# 进行四倍超采样
def upsample_esf(esf):
    # 对ESF进行四倍超采样
    upsampled_esf = np.interp(np.linspace(0, len(esf) - 1, len(esf) * 4), np.arange(len(esf)), esf)
    return upsampled_esf


# 通过差分运算获得LSF
def compute_lsf(esf):
    # 使用差分计算LSF
    lsf = np.diff(esf)
    return lsf


# 对LSF应用汉明窗
def apply_hamming_window(lsf):
    # 使用汉明窗进行加窗处理
    hamming_window = np.hamming(len(lsf))
    return lsf * hamming_window


# 进行DFT运算
def compute_sfr(lsf, sampling_frequency):
    # 对LSF进行傅里叶变换，计算空间频率响应（SFR）
    sfr = np.abs(fftshift(fft(lsf)))

    # 计算频率轴
    nyquist_frequency = sampling_frequency / 2
    frequencies = fftshift(fftfreq(len(lsf), 1 / sampling_frequency))

    # 只保留正频率部分
    positive_frequencies = frequencies[frequencies >= 0]
    sfr_positive = sfr[frequencies >= 0]

    return positive_frequencies, sfr_positive


# 计算平均SFR
def compute_average_sfr(frequencies, sfr, freq_min=0, freq_max=None):
    if freq_max is None:
        freq_max = max(frequencies)  # 默认最大频率为奈奎斯特频率

    # 选择在指定范围内的频率
    selected_freqs = (frequencies >= freq_min) & (frequencies <= freq_max)

    # 计算指定频率范围内的平均SFR
    average_sfr = np.mean(sfr[selected_freqs])
    return average_sfr, frequencies[selected_freqs], sfr[selected_freqs]


# 计算目标MTF
def target_mtf(frequencies):
    return 1 - 0.1030396 * frequencies + 0.00422688 * frequencies ** 2 - 0.0000915521 * frequencies ** 3 + 0.00000103079 * frequencies ** 4


# 计算扫描仪MTF
def compute_scanner_mtf(target_mtf_values, system_mtf_values):
    return target_mtf_values / system_mtf_values


# 计算MTF补偿后的图像
def compensate_mtf(image, target_mtf_values, scanner_mtf):
    # 计算2D傅里叶变换
    fft_image = fftshift(fft2(image))

    # 计算频率值
    nyquist_frequency = 1 / 2  # 假设采样频率为1，修改根据实际情况
    freq_x, freq_y = np.meshgrid(fftfreq(image.shape[1], 1 / nyquist_frequency),
                                 fftfreq(image.shape[0], 1 / nyquist_frequency))

    # 计算MTF补偿因子
    mtf_compensation_factor = target_mtf(np.sqrt(freq_x ** 2 + freq_y ** 2)) * scanner_mtf

    # 应用MTF补偿
    compensated_image = np.real(ifft2(fft_image / mtf_compensation_factor))

    return compensated_image, freq_x, freq_y


# 主程序
def main():
    # 加载图像
    image_path = 'ES10000005_1200_Image2_1_MTF.tif'  # 请替换为实际的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError("无法读取图像文件，请检查文件路径")

    # 获取ROI
    roi = get_roi(image, x_start=100, y_start=50, width=200, height=150)  # 示例，调整为实际ROI的位置和大小

    # 计算R、G、B通道的ESF
    esf_r = compute_esf(roi[:, :, 0])  # R通道
    esf_g = compute_esf(roi[:, :, 1])  # G通道
    esf_b = compute_esf(roi[:, :, 2])  # B通道

    # 对ESF进行四倍超采样
    upsampled_esf_r = upsample_esf(esf_r)
    upsampled_esf_g = upsample_esf(esf_g)
    upsampled_esf_b = upsample_esf(esf_b)

    # 计算LSF
    lsf_r = compute_lsf(upsampled_esf_r)
    lsf_g = compute_lsf(upsampled_esf_g)
    lsf_b = compute_lsf(upsampled_esf_b)

    # 对LSF应用汉明窗
    lsf_r_windowed = apply_hamming_window(lsf_r)
    lsf_g_windowed = apply_hamming_window(lsf_g)
    lsf_b_windowed = apply_hamming_window(lsf_b)

    # 计算SFR
    sampling_frequency = 1  # 示例值，修改为实际采样频率
    freqs_r, sfr_r = compute_sfr(lsf_r_windowed, sampling_frequency)
    freqs_g, sfr_g = compute_sfr(lsf_g_windowed, sampling_frequency)
    freqs_b, sfr_b = compute_sfr(lsf_b_windowed, sampling_frequency)

    # 计算平均SFR
    avg_sfr_r, selected_freqs_r, avg_sfr_values_r = compute_average_sfr(freqs_r, sfr_r)
    avg_sfr_g, selected_freqs_g, avg_sfr_values_g = compute_average_sfr(freqs_g, sfr_g)
    avg_sfr_b, selected_freqs_b, avg_sfr_values_b = compute_average_sfr(freqs_b, sfr_b)

    # 打印平均SFR
    print(f"R通道平均SFR: {avg_sfr_r}")
    print(f"G通道平均SFR: {avg_sfr_g}")
    print(f"B通道平均SFR: {avg_sfr_b}")

    # 计算目标MTF和扫描仪MTF
    system_mtf = np.ones_like(freqs_r)  # 假设系统MTF为1，实际应根据测量数据来获取
    target_mtf_values = target_mtf(freqs_r)
    scanner_mtf = compute_scanner_mtf(target_mtf_values, system_mtf)

    # 计算MTF补偿后的图像
    compensated_image, freq_x, freq_y = compensate_mtf(roi, target_mtf_values, scanner_mtf)

    # 显示结果
    plt.figure(figsize=(10, 6))
    plt.plot(selected_freqs_r, avg_sfr_values_r, label='R通道 SFR')
    plt.plot(selected_freqs_g, avg_sfr_values_g, label='G通道 SFR')
    plt.plot(selected_freqs_b, avg_sfr_values_b, label='B通道 SFR')
    plt.xlabel('空间频率 (每像素周期)')
    plt.ylabel('SFR')
    plt.title('各通道的空间频率响应（SFR）')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 显示补偿后的图像
    plt.imshow(compensated_image, cmap='gray')
    plt.title("MTF补偿后的图像")
    plt.show()


if __name__ == "__main__":
    main()
