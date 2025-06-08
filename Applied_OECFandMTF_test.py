import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.interpolate import interp1d
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib import rcParams
import pandas as pd
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 设置Matplotlib后端
matplotlib.use('TkAgg')


def load_oecf_data(oecf_path):
    """加载三通道OECF数据并生成正向和反向插值函数"""
    data = pd.read_excel(oecf_path).values

    # 添加数据验证
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("OECF文件格式错误：应包含R、G、B三列数据")

    # 检查并移除重复数据点
    data = np.unique(data, axis=0)

    # 确保数据按升序排列
    sort_idx = np.argsort(data[:, 0])
    data = data[sort_idx]

    reflectance = np.linspace(0, 1, len(data))

    # 创建插值函数
    f_r = interp1d(reflectance, data[:, 0], kind='quadratic',
                   bounds_error=False, fill_value=(data[0, 0], data[-1, 0]))
    f_g = interp1d(reflectance, data[:, 1], kind='quadratic',
                   bounds_error=False, fill_value=(data[0, 1], data[-1, 1]))
    f_b = interp1d(reflectance, data[:, 2], kind='quadratic',
                   bounds_error=False, fill_value=(data[0, 2], data[-1, 2]))

    # 生成256级插值结果
    reflectance_256 = np.linspace(0, 1, 256)
    oecf_r = f_r(reflectance_256)
    oecf_g = f_g(reflectance_256)
    oecf_b = f_b(reflectance_256)

    # 绘制原始数据点和插值曲线
    plt.figure(figsize=(10, 6))

    # 绘制原始数据点
    plt.scatter(reflectance, data[:, 0], color='red', marker='o', label='原始R通道数据点')
    plt.scatter(reflectance, data[:, 1], color='green', marker='s', label='原始G通道数据点')
    plt.scatter(reflectance, data[:, 2], color='blue', marker='^', label='原始B通道数据点')

    # 绘制插值曲线
    plt.plot(reflectance_256, oecf_r, 'r-', alpha=0.7, label='R通道插值曲线')
    plt.plot(reflectance_256, oecf_g, 'g-', alpha=0.7, label='G通道插值曲线')
    plt.plot(reflectance_256, oecf_b, 'b-', alpha=0.7, label='B通道插值曲线')

    plt.title('OECF数据点与插值曲线对比')
    plt.xlabel('反射率')
    plt.ylabel('数字值')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (oecf_r, oecf_g, oecf_b)


def load_mtf_data(mtf_path, is_scanner=False):
    """改进的MTF数据加载函数"""
    try:
        # 自动检测文件是否有表头
        with open(mtf_path, encoding='utf-8') as f:  # 明确指定UTF-8编码
            first_line = f.readline().strip()
            has_header = any(c.isalpha() for c in first_line)

        data = np.loadtxt(mtf_path, skiprows=1 if has_header else 0)

        if is_scanner:
            # 扫描仪MTF文件格式：SF, Y_H, Y_V, G_H, G_V
            return {
                'freq': data[:, 0],
                'y_h': data[:, 1],
                'y_v': data[:, 2],
                'g_h': data[:, 3],
                'g_v': data[:, 4]
            }
        else:
            # 目标MTF文件格式：SF, G, Y
            return {
                'freq': data[:, 0],
                'g_mtf': data[:, 1],
                'y_mtf': data[:, 2]
            }
    except Exception as e:
        print(f"加载MTF数据错误: {str(e)}")
        print(f"问题文件: {mtf_path}")
        return None


def create_mtf_compensation_filter(image_shape, scanner_mtf, target_mtf, dpi=1200):
    """专为字符测量优化的MTF补偿滤波器"""
    rows, cols = image_shape
    pixel_size = 25.4 / dpi  # mm/pixel

    # 只需处理水平方向频率
    u = np.fft.fftfreq(cols) * (1 / pixel_size)
    v = np.zeros(rows)  # 垂直方向不处理
    uu, vv = np.meshgrid(u, v)

    # 目标MTF插值（仅Y通道）
    target_interp_y = interp1d(target_mtf['freq'], target_mtf['y_mtf'],
                               bounds_error=False, fill_value=0.0)

    # 扫描仪MTF插值（仅Y水平）
    scanner_interp_y = interp1d(scanner_mtf['freq'], scanner_mtf['y_h'],
                                bounds_error=False, fill_value=0.0)

    # 计算补偿比（仅水平方向）
    freq = np.abs(uu)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(scanner_interp_y(freq) > 0.01,
                         target_interp_y(freq) / scanner_interp_y(freq),
                         1.0)

    # 限制补偿幅度
    return np.clip(ratio, 0.5, 2.0)  # 更保守的范围防止过补偿


def apply_frequency_domain_compensation(img, oecf_r, oecf_g, oecf_b, mtf_filter):
    """优化后的字符线条补偿"""
    # 1. 线性化处理（仅需Y通道）
    inv_oecf_r = interp1d(oecf_r, np.linspace(0, 1, 256),
                          bounds_error=False, fill_value=(0, 1))
    linear_y = 0.299 * inv_oecf_r(img[..., 0]) + \
               0.587 * inv_oecf_r(img[..., 1]) + \
               0.114 * inv_oecf_r(img[..., 2])

    # 2. 仅应用水平方向补偿
    fft_y = np.fft.fft(linear_y, axis=1)
    fft_y_comp = fft_y * fftshift(mtf_filter, axes=1)

    # 3. 逆变换
    restored_y = np.real(np.fft.ifft(fft_y_comp, axis=1))

    # 4. 保持色度不变，仅调整亮度
    restored = img.copy()
    delta = restored_y - linear_y
    for c in range(3):
        restored[..., c] = np.clip(img[..., c] + delta, 0, 1)

    return restored


def process_image_restoration(input_path, output_path, oecf_data, scanner_mtf, target_mtf, dpi=1200):
    """完整的图像还原流程"""
    # 加载图像
    pil_img = Image.open(input_path)
    img_array = np.array(pil_img)

    # 处理图像类型
    if img_array.dtype == np.uint16:
        img_float = img_array.astype(np.float32) / 65535.0
        output_scale = 65535
    elif img_array.dtype == np.uint8:
        img_float = img_array.astype(np.float32) / 255.0
        output_scale = 255
    else:
        raise ValueError(f"Unsupported image dtype: {img_array.dtype}")

    # 确保是彩色图像
    if len(img_array.shape) == 2:
        img_float = np.stack([img_float] * 3, axis=-1)

    # 创建MTF补偿滤波器
    mtf_filter = create_mtf_compensation_filter(
        img_float.shape[:2],
        scanner_mtf,
        {'freq': target_mtf['freq'], 'y_mtf': target_mtf['y_mtf']},  # 只传Y通道
        dpi
    )

    # 应用补偿（字符测量优化版）
    restored = apply_frequency_domain_compensation(
        img_float,
        oecf_data[0], oecf_data[1], oecf_data[2],
        mtf_filter  # 单通道滤波器
    )

    # 确保没有无效值
    restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)

    # 转换回原始位深
    output_img = (restored * output_scale).clip(0, output_scale).astype(img_array.dtype)

    # 保存结果
    Image.fromarray(output_img).save(output_path)
    return output_img


# 主程序
if __name__ == "__main__":
    # 文件路径
    oecf_file = 'D:/PycharmProjects/pythonProject/reflectance_values_excel.xlsx'
    scanner_mtf_file = 'C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/MTFfile/file2-1.csv'
    target_mtf_file = 'C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/data/Target_MTF_1200_2021.txt'
    input_image = 'C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000005_1200_Image2_1.tif'
    output_image = 'ES10000005_1200_Image2_1_applied_freq.tif'

    try:
        # 1. 加载所有必需数据
        oecf_data = load_oecf_data(oecf_file)
        scanner_mtf = load_mtf_data(scanner_mtf_file, is_scanner=True)
        target_mtf = load_mtf_data(target_mtf_file, is_scanner=False)

        # 2. 验证数据加载是否成功
        if oecf_data is None:
            raise ValueError("无法加载OECF数据")
        if scanner_mtf is None or target_mtf is None:
            raise ValueError("无法加载MTF数据")

        # 3. 绘制MTF对比曲线
        plt.figure(figsize=(12, 5))
        plt.plot(scanner_mtf['freq'], scanner_mtf['y_h'], 'r-', label='Scanner Y Horizontal')
        plt.plot(scanner_mtf['freq'], scanner_mtf['y_v'], 'r--', label='Scanner Y Vertical')
        plt.plot(target_mtf['freq'], target_mtf['y_mtf'], 'b-', label='Target Y')
        plt.plot(target_mtf['freq'], target_mtf['g_mtf'], 'g-', label='Target G')
        plt.title('MTF对比曲线')
        plt.xlabel('空间频率 (cycles/mm)')
        plt.ylabel('MTF值')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 4. 执行图像还原
        result = process_image_restoration(
            input_image,
            output_image,
            oecf_data,
            scanner_mtf,
            target_mtf,
            dpi=1200
        )

        print(f"还原完成，结果保存到: {output_image}")

        # 5. 显示对比结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        original = Image.open(input_image)
        axes[0].imshow(original)
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(Image.fromarray(result))
        axes[1].set_title('频域补偿后图像')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")