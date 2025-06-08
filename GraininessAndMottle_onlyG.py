import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.filters import gaussian, difference_of_gaussians
import json
from datetime import datetime
import pandas as pd
import pywt

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 常量定义
MICRONS_PER_INCH = 25400
DEFAULT_DPI = 1200
BLOCK_SIZE = 81  # 分块数量
MOTTLE_CUTOFF_FREQ = 0.4  # cycles/mm


def load_oecf_data(oecf_path):
    """改进的OECF加载函数，确保正确应用曲线"""
    try:
        # 使用pandas读取更可靠
        df = pd.read_excel(oecf_path, header=None)

        # 确保有足够的数据点
        if len(df) < 256:
            raise ValueError("OECF曲线需要至少256个数据点")

        # 提取RGB通道
        oecf_r = df.iloc[:256, 0].values.astype(np.float32)
        oecf_g = df.iloc[:256, 1].values.astype(np.float32)
        oecf_b = df.iloc[:256, 2].values.astype(np.float32)

        # 验证曲线有效性
        if np.any(oecf_r < 0) or np.any(oecf_r > 1) or \
                np.any(oecf_g < 0) or np.any(oecf_g > 1) or \
                np.any(oecf_b < 0) or np.any(oecf_b > 1):
            raise ValueError("OECF曲线值必须在0-1范围内")

        return [oecf_r, oecf_g, oecf_b]

    except Exception as e:
        raise ValueError(f"OECF文件加载失败: {str(e)}")


def verify_oecf(oecf_data):
    """验证OECF曲线有效性"""
    if len(oecf_data) != 3 or any(len(channel) != 256 for channel in oecf_data):
        raise ValueError("无效的OECF数据: 需要RGB三通道且每通道256级")
    return True


def apply_oecf(image, oecf_data, measurement_type="banding"):
    """
    应用OECF校准，根据测量类型选择正确的转换方式

    Args:
        measurement_type:
            "banding" - 使用sRGB转换（用于banding/mottle/graininess）
            "line" - 使用Rvis转换（用于字符和线条测量）
    """
    # 确保图像是8位或16位
    if image.dtype == np.uint16:
        img_float = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # 验证OECF数据
    verify_oecf(oecf_data)

    # 应用OECF到每个通道
    calibrated = np.zeros_like(img_float)
    for c in range(3):
        lut = np.interp(np.linspace(0, 1, 256),
                        np.linspace(0, 1, len(oecf_data[c])),
                        oecf_data[c])
        channel = np.clip(img_float[..., c] * 255, 0, 255).astype(np.uint8)
        calibrated[..., c] = lut[channel]

    # 根据测量类型选择转换方式
    if measurement_type == "banding":
        # 使用标准sRGB转换公式 (IEC 61966-2-1)
        calibrated_y = 0.2126 * calibrated[..., 0] + \
                       0.7152 * calibrated[..., 1] + \
                       0.0722 * calibrated[..., 2]
    else:
        # 使用Rvis转换 (ISO 24790 第5.2.5节)
        calibrated_y = 0.93137 * calibrated[..., 0] + \
                       0.933 * calibrated[..., 1] + \
                       0.933 * calibrated[..., 2]

    return np.clip(calibrated_y, 0, 1)


def calibrate_reflectance(image, oecf_path=None, dpi=DEFAULT_DPI, measurement_type="banding"):
    """
    反射率校准函数（仅返回G通道，不转换CIE Y）
    """
    if oecf_path:
        oecf_data = load_oecf_data(oecf_path)
        calibrated = apply_oecf(image, oecf_data, measurement_type)
        # 仅返回G通道（索引1）
        if len(calibrated.shape) > 2:  # 如果是彩色图像
            return calibrated[..., 1]  # G通道
        return calibrated
    else:
        # 无OECF文件时，直接返回G通道
        img_float = image.astype(np.float32) / 255.0
        if len(img_float.shape) > 2:  # 如果是彩色图像
            return img_float[..., 1]  # G通道
        return img_float


def partition_image(image, num_blocks=BLOCK_SIZE):
    """将图像分割为N×N的均匀分块"""
    h, w = image.shape[:2]
    block_h, block_w = h // num_blocks, w // num_blocks
    blocks_list = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            y1 = i * block_h
            y2 = (i + 1) * block_h if i < num_blocks - 1 else h
            x1 = j * block_w
            x2 = (j + 1) * block_w if j < num_blocks - 1 else w
            blocks_list.append(image[y1:y2, x1:x2])

    return blocks_list


def measure_graininess(roi_image, dpi=DEFAULT_DPI):
    """
    测量微观非周期性亮度波动（简化通道处理）
    """
    # 如果图像是多通道，仅提取G通道
    if len(roi_image.shape) > 2:
        roi_image = roi_image[..., 1]  # G通道索引为1

    # 检查ROI尺寸 (最小600x600像素@1200dpi)
    min_pixels = int(12.7 * dpi / 25.4)
    if roi_image.shape[0] < min_pixels or roi_image.shape[1] < min_pixels:
        raise ValueError(f"ROI for graininess too small. Minimum {min_pixels}x{min_pixels} pixels at {dpi} dpi")

    # 计算合适的小波分解层级
    max_level = pywt.dwt_max_level(min(roi_image.shape), 'db16')
    level = min(4, max_level)  # 限制最大层级为4

    # 应用小波变换 (Daubechies 16阶)
    coeffs = pywt.wavedec2(roi_image, 'db16', level=level)

    # 保留1.4763-0.3691 cy/mm频率 (小波层1和0)
    # 置零高层小波系数(层2及以上)和近似系数(层0)
    for i in range(2, level + 1):
        coeffs[i] = tuple([np.zeros_like(v) for v in coeffs[i]])
    coeffs[0] = np.zeros_like(coeffs[0])  # 近似系数

    # 逆小波变换
    filtered = pywt.waverec2(coeffs, 'db16')

    # 裁剪边缘 (30像素)
    crop_size = int(0.635 * dpi / 25.4)  # 0.635mm
    filtered = filtered[crop_size:-crop_size, crop_size:-crop_size]

    # 分块计算方差 (至少81块，每块至少60x60像素)
    block_size = int(1.27 * dpi / 25.4)  # 1.27mm
    h_blocks = filtered.shape[0] // block_size
    v_blocks = filtered.shape[1] // block_size

    variances = []
    for i in range(h_blocks):
        for j in range(v_blocks):
            block = filtered[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            variances.append(np.var(block))

    return np.sqrt(np.mean(variances)) if variances else 0.0


def measure_mottle(roi_image, dpi=DEFAULT_DPI):
    """
    测量宏观非周期性亮度波动（简化通道处理）
    """
    # 如果图像是多通道，仅提取G通道
    if len(roi_image.shape) > 2:
        roi_image = roi_image[..., 1]  # G通道索引为1

    # 检查ROI尺寸 (最小1200x1200像素@1200dpi)
    min_pixels = int(25.4 * dpi / 25.4)
    if roi_image.shape[0] < min_pixels or roi_image.shape[1] < min_pixels:
        raise ValueError(f"ROI for mottle too small. Minimum {min_pixels}x{min_pixels} pixels at {dpi} dpi")

    # 计算合适的小波分解层级
    max_level = pywt.dwt_max_level(min(roi_image.shape), 'db16')
    level = min(6, max_level)  # 限制最大层级为6

    # 应用小波变换 (Daubechies 16阶)
    coeffs = pywt.wavedec2(roi_image, 'db16', level=level)

    # 保留0.3691-0.0461 cy/mm频率 (小波层2,1,0)
    # 置零高层小波系数(层3及以上)和近似系数(层0)
    for i in range(3, level + 1):
        coeffs[i] = tuple([np.zeros_like(v) for v in coeffs[i]])
    coeffs[0] = np.zeros_like(coeffs[0])  # 近似系数

    # 逆小波变换
    filtered = pywt.waverec2(coeffs, 'db16')

    # 裁剪边缘 (60像素)
    crop_size = int(1.27 * dpi / 25.4)  # 1.27mm
    filtered = filtered[crop_size:-crop_size, crop_size:-crop_size]

    # 分块计算方差 (至少81块，每块至少120x120像素)
    block_size = int(2.54 * dpi / 25.4)  # 2.54mm
    h_blocks = filtered.shape[0] // block_size
    v_blocks = filtered.shape[1] // block_size

    variances = []
    for i in range(h_blocks):
        for j in range(v_blocks):
            block = filtered[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            variances.append(np.var(block))

    return np.sqrt(np.mean(variances)) if variances else 0.0


def measure_texture_attributes(image_path, dpi=DEFAULT_DPI, oecf_path=None, debug=False, measurement_type="banding"):
    """
    纹理属性测量主函数

    Args:
        measurement_type: "banding"或"line"，指定OECF校准模式
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB顺序

    calibrated = calibrate_reflectance(img, oecf_path=oecf_path, dpi=dpi, measurement_type=measurement_type)
    h, w = calibrated.shape[:2]

    # 调整ROI区域大小，确保足够大
    # 对于Mottle测量，需要至少1200x1200像素@1200dpi
    min_roi_size = int(25.4 * dpi / 25.4)  # 25.4mm @ 1200dpi = 1200px

    # 如果图像太小，使用整个图像作为ROI
    if h < min_roi_size or w < min_roi_size:
        roi = calibrated
        print(f"警告: 图像尺寸({w}x{h})小于最小ROI要求({min_roi_size}x{min_roi_size})，将使用整个图像")
    else:
        # 否则使用中心区域，确保足够大
        roi = calibrated[h // 2 - min_roi_size // 2: h // 2 + min_roi_size // 2,
              w // 2 - min_roi_size // 2: w // 2 + min_roi_size // 2]

    results = {
        "Graininess": measure_graininess(roi, dpi),
        "Mottle": measure_mottle(roi, dpi)
    }

    if debug:
        visualize_texture_results(roi, results)

    return results


def visualize_texture_results(roi_image, results):
    """纹理测量结果可视化"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(roi_image, cmap='gray', vmin=0, vmax=1)
    plt.title("ROI区域")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    graininess_map = difference_of_gaussians(roi_image, 1, 1.6)
    plt.imshow(graininess_map, cmap='gray')
    plt.title(f"Graininess: {results['Graininess']:.4f}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    mottle_map = gaussian(roi_image, sigma=2)
    plt.imshow(mottle_map, cmap='gray')
    plt.title(f"Mottle: {results['Mottle']:.4f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_texture_results(results, output_path=None):
    """保存纹理测量结果"""
    converted_results = {
        key: float(value) if isinstance(value, (np.float32, np.float64)) else value
        for key, value in results.items()
    }

    data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_standard": "ISO/IEC 24790 Texture Metrics",
            "software": "texture_metrics.py"
        },
        "texture_metrics": {
            "graininess": converted_results["Graininess"],
            "mottle": converted_results["Mottle"]
        }
    }

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"texture_metrics_results_{timestamp}"

    json_path = f"{output_path}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    excel_data = [
        ["Category", "Metric", "Value"],
        ["Texture", "Graininess", converted_results["Graininess"]],
        ["Texture", "Mottle", converted_results["Mottle"]]
    ]

    df = pd.DataFrame(excel_data[1:], columns=excel_data[0])
    df.to_excel(f"{output_path}.xlsx", index=False)


if __name__ == "__main__":
    test_image_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/2_1cut/ES10000005_1200_Image2_1_LA18_Graininess.tif"
    oecf_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/OECFfile/image2_2.xlsx"  # 替换为实际的OECF文件路径

    results = measure_texture_attributes(
        test_image_path,
        dpi=1200,
        oecf_path=oecf_path,
        debug=True,
        measurement_type="banding"  # 明确指定使用banding模式
    )

    save_texture_results(results)

    print("\n纹理测量结果:")
    print(f"微观纹理波动 (Graininess): {results['Graininess']:.4f}")
    print(f"宏观纹理波动 (Mottle): {results['Mottle']:.4f}")