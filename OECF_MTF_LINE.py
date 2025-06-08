import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.interpolate import interp1d
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib import rcParams
from matplotlib.widgets import RectangleSelector
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')

# 全局变量存储ROI坐标
roi_coords = None
current_image = None


def onselect(eclick, erelease):
    """矩形选择回调函数"""
    global roi_coords
    roi_coords = (int(eclick.xdata), int(eclick.ydata),
                  int(erelease.xdata), int(erelease.ydata))
    print(f"已选择ROI区域: {roi_coords}")


def select_roi_interactive(image):
    """交互式选择ROI区域"""
    global current_image
    current_image = image.copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("请用鼠标框选ROI区域")

    # 创建矩形选择器
    rs = RectangleSelector(ax, onselect,
                           useblit=True,
                           button=[1],  # 左键选择
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)

    plt.show()
    return roi_coords


def load_oecf_data(oecf_path):
    """加载OECF数据"""
    data = np.loadtxt(oecf_path)
    r_val, r_refl = data[:, 0], np.clip(data[:, 1], 0.001, 0.933)
    g_val, g_refl = data[:, 0], np.clip(data[:, 2], 0.001, 0.933)
    b_val, b_refl = data[:, 0], np.clip(data[:, 3], 0.001, 0.933)
    return r_val, r_refl, g_val, g_refl, b_val, b_refl


def load_mtf_data(mtf_path, is_scanner=False):
    """加载MTF数据"""
    try:
        with open(mtf_path) as f:
            first_line = f.readline().strip()
            has_header = any(c.isalpha() for c in first_line)

        data = np.loadtxt(mtf_path, skiprows=1 if has_header else 0)

        if is_scanner:
            return {
                'freq': data[:, 0],
                'y_h': data[:, 1],
                'y_v': data[:, 2],
                'g_h': data[:, 3],
                'g_v': data[:, 4]
            }
        else:
            return {
                'freq': data[:, 0],
                'g_mtf': data[:, 1],
                'y_mtf': data[:, 2]
            }
    except Exception as e:
        print(f"加载MTF数据错误: {str(e)}")
        return None


def create_mtf_compensation_filter(image_shape, scanner_mtf, target_mtf, dpi=1200):
    """创建MTF补偿滤波器"""
    rows, cols = image_shape
    pixel_size = 25.4 / dpi  # 计算像素大小（毫米）

    # 生成频率网格
    u = np.fft.fftfreq(cols) * cols  # 水平频率
    v = np.fft.fftfreq(rows) * rows  # 垂直频率
    uu, vv = np.meshgrid(u, v)
    rho = np.sqrt(uu ** 2 + vv ** 2) / cols  # 归一化空间频率
    rho_mm = rho * (1 / pixel_size)  # 转换为cycles/mm

    # 高频截断（0.5 cycles/mm）
    mask = rho_mm <= 0.5

    # 创建插值函数
    scanner_h = interp1d(scanner_mtf['freq'], scanner_mtf['y_h'],
                         bounds_error=False, fill_value=0.0)
    scanner_v = interp1d(scanner_mtf['freq'], scanner_mtf['y_v'],
                         bounds_error=False, fill_value=0.0)
    target_interp = interp1d(target_mtf['freq'], target_mtf['y_mtf'],
                             bounds_error=False, fill_value=0.0)

    # 计算补偿比（分方向平均）
    scanner_combined = 0.5 * (scanner_h(rho_mm) + scanner_v(rho_mm))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(mask,
                         target_interp(rho_mm) / scanner_combined,
                         0.0)

    return np.clip(ratio, 0.5, 3.0)  # 限制增益范围


def restore_original_image(scan_img, device_r_lut, device_g_lut, device_b_lut, mtf_filter):
    """应用OECF和MTF补偿"""
    restored = np.zeros_like(scan_img)
    restored[..., 0] = np.interp(scan_img[..., 0], np.linspace(0, 1, 256), device_r_lut)
    restored[..., 1] = np.interp(scan_img[..., 1], np.linspace(0, 1, 256), device_g_lut)
    restored[..., 2] = np.interp(scan_img[..., 2], np.linspace(0, 1, 256), device_b_lut)

    # 应用MTF补偿
    for c in range(3):
        fft_channel = fftshift(fft2(restored[..., c]))
        fft_compensated = fft_channel * mtf_filter
        restored[..., c] = np.clip(np.real(ifft2(ifftshift(fft_compensated))), 0, 1)

    return restored


def process_image_restoration(input_path, oecf_data, scanner_mtf, target_mtf, dpi=1200):
    """完整的图像还原流程"""
    pil_img = Image.open(input_path)
    img_array = np.array(pil_img)

    if img_array.dtype == np.uint16:
        img_float = img_array.astype(np.float32) / 65535.0
    elif img_array.dtype == np.uint8:
        img_float = img_array.astype(np.float32) / 255.0
    else:
        raise ValueError(f"不支持的图像格式: {img_array.dtype}")

    if len(img_array.shape) == 2:
        img_float = np.stack([img_float] * 3, axis=-1)

    # 构建设备特性LUT和MTF补偿滤波器
    device_r, device_g, device_b = [build_device_lut(c) for c in oecf_data]
    mtf_filter = create_mtf_compensation_filter(img_float.shape[:2], scanner_mtf, target_mtf, dpi)

    # 还原原始图像
    restored = restore_original_image(img_float, device_r, device_g, device_b, mtf_filter)

    if img_array.dtype == np.uint16:
        output_img = (restored * 65535).clip(0, 65535).astype(np.uint16)
    else:
        output_img = (restored * 255).clip(0, 255).astype(np.uint8)

    return output_img


def build_device_lut(curve):
    """构建设备响应LUT"""
    return curve


def measure_line_attributes(roi_image, dpi=1200):
    """测量线条属性"""
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # 检测线条方向
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        raise ValueError("未检测到线条")

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    avg_angle = np.mean(angles)
    orientation = 'horizontal' if abs(avg_angle) < 45 else 'vertical'

    # 计算反射率直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peaks, _ = find_peaks(hist.flatten(), height=50, distance=50)

    if len(peaks) < 2:
        raise ValueError("反射率分布不呈现有效的双峰形状")

    r_min = peaks[0] / 255.0
    r_max = peaks[1] / 255.0

    # 计算各种阈值
    r10 = r_min + 0.1 * (r_max - r_min)
    r40 = r_min + 0.4 * (r_max - r_min)
    r70 = r_min + 0.7 * (r_max - r_min)

    # 测量线宽
    _, binary = cv2.threshold(gray, int(r40 * 255), 255, cv2.THRESH_BINARY_INV)
    line_width_px = measure_line_width(binary, num_samples=10)
    line_width_mm = (line_width_px / dpi) * 25.4

    # 计算线条图像密度
    r25 = r_min + 0.25 * (r_max - r_min)
    line_density = -np.log10(r25)

    # 计算字符暗度
    character_darkness = line_density * np.sqrt(line_width_mm)

    return {
        "R_max": r_max,
        "R_min": r_min,
        "R10": r10,
        "R40": r40,
        "R70": r70,
        "LineWidth_mm": line_width_mm,
        "LineDensity": line_density,
        "CharacterDarkness": character_darkness,
        "Orientation": orientation
    }


def measure_line_width(binary_image, num_samples=10):
    """测量线宽"""
    height, width = binary_image.shape
    line_widths = []

    for i in range(num_samples):
        y = int(height * (i + 1) / (num_samples + 1))
        profile = binary_image[y, :]
        left_edge = np.argmax(profile > 0)
        right_edge = len(profile) - np.argmax(profile[::-1] > 0) - 1
        line_widths.append(right_edge - left_edge)

    return np.mean(line_widths)


def main():
    # 文件路径
    oecf_file = 'path/to/oecf.csv'
    scanner_mtf_file = 'path/to/scanner_mtf.csv'
    target_mtf_file = 'path/to/target_mtf.csv'
    input_image = 'path/to/input_image.tif'

    # 加载数据
    oecf_data = load_oecf_data(oecf_file)
    scanner_mtf = load_mtf_data(scanner_mtf_file, is_scanner=True)
    target_mtf = load_mtf_data(target_mtf_file, is_scanner=False)

    if None in [oecf_data, scanner_mtf, target_mtf]:
        raise ValueError("无法加载必要的数据文件")

    # 执行图像还原
    restored_image = process_image_restoration(
        input_image, oecf_data, scanner_mtf, target_mtf, dpi=1200
    )

    # 显示还原后的图像并选择ROI
    roi_coords = select_roi_interactive(restored_image)
    if roi_coords is None:
        raise ValueError("未选择ROI区域")

    # 提取ROI区域
    x1, y1, x2, y2 = roi_coords
    roi_image = restored_image[y1:y2, x1:x2]

    # 测量线条属性
    results = measure_line_attributes(roi_image, dpi=1200)

    # 显示结果
    print("\n测量结果:")
    print(f"背景最大反射率 (R_max): {results['R_max']:.4f}")
    print(f"线条最小反射率 (R_min): {results['R_min']:.4f}")
    print(f"内边界阈值 (R10): {results['R10']:.4f}")
    print(f"边缘阈值 (R40): {results['R40']:.4f}")
    print(f"外边界阈值 (R70): {results['R70']:.4f}")
    print(f"线宽: {results['LineWidth_mm']:.5f} mm")
    print(f"线条图像密度: {results['LineDensity']:.4f}")
    print(f"字符暗度: {results['CharacterDarkness']:.5f}")
    print(f"线条方向: {results['Orientation']}")

    # 显示ROI区域
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    plt.title("ROI区域")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()