import pandas as pd
from io import StringIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks
import json
from datetime import datetime
import pandas as pd
import openpyxl
import math

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 常量定义
DEFAULT_DPI = 600  # 根据ISO标准使用600ppi
MM_TO_INCH = 25.4

def mm_to_px(mm, dpi=DEFAULT_DPI):
    """毫米到像素的转换"""
    return int(round(mm * dpi / MM_TO_INCH))


def parse_config_file(file_content):
    config_data = {
        'Target': {},
        'Fiducials': {},
        'Measurement': {}
    }

    sections = file_content.split('\n\n')

    for section in sections:
        lines = [line for line in section.strip().split('\n') if line.strip()]
        if not lines:
            continue

        section_name = lines[0].split('\t')[0].strip()

        if section_name == 'Fiducials':
            # 跳过标题行（"ID X Y"）
            for line in lines[1:]:
                parts = [p.strip() for p in line.split('\t') if p.strip()]

                # 处理带序号的行（如"4 ULC 7.62 10.92 ..."）
                if len(parts) >= 4 and parts[1] in ["ULC", "URC", "LRC", "LLC"]:
                    fiducial_id = parts[1]
                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                        config_data['Fiducials'][fiducial_id] = (x, y)
                    except (ValueError, IndexError):
                        continue

                # 处理不带序号的行（如"URC 174.16 10.92"）
                elif len(parts) >= 3 and parts[0] in ["ULC", "URC", "LRC", "LLC"]:
                    fiducial_id = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        config_data['Fiducials'][fiducial_id] = (x, y)
                    except (ValueError, IndexError):
                        continue

        elif section_name == 'Target':
            if len(lines) > 1:
                config_data['Target']['Name'] = lines[1].split('\t')[1].strip()

        elif section_name == 'Measurement':
            for line in lines[1:]:
                parts = [p.strip() for p in line.split('\t')[:6] if p.strip()]
                if len(parts) >= 6:
                    measurement_id = parts[1]
                    try:
                        config_data['Measurement'][measurement_id] = {
                            'X': float(parts[2]),
                            'Y': float(parts[3]),
                            'dX': float(parts[4]),
                            'dY': float(parts[5])
                        }
                    except (ValueError, IndexError):
                        continue

    return config_data


def load_oecf_data_from_excel(oecf_path):
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


def calibrate_reflectance(image, oecf_path=None):
    """改进的反射率校准函数，确保正确应用OECF"""
    # 1. 加载OECF数据（必须提供）
    if oecf_path is None:
        raise ValueError("必须提供OECF曲线文件")

    oecf_data = load_oecf_data_from_excel(oecf_path)
    verify_oecf(oecf_data)

    # 2. 图像预处理（保持原始位深度）
    if image.dtype == np.uint16:
        img_float = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # 3. 应用OECF校准（确保使用插值）
    calibrated = np.zeros_like(img_float)
    for c in range(3):
        # 创建查找表（LUT）
        lut = np.zeros(256 if image.dtype == np.uint8 else 65536, dtype=np.float32)
        x_original = np.linspace(0, 1, len(oecf_data[c]))
        lut = np.interp(np.linspace(0, 1, len(lut)), x_original, oecf_data[c])

        # 应用LUT
        if image.ndim == 2:  # 灰度图像
            channel = np.clip(img_float * (len(lut) - 1), 0, len(lut) - 1).astype(np.uint16)
            calibrated = lut[channel]
        else:  # 彩色图像
            channel = np.clip(img_float[..., c] * (len(lut) - 1), 0, len(lut) - 1).astype(np.uint16)
            calibrated[..., c] = lut[channel]

    # 4. 转换为Y通道反射率
    if calibrated.ndim == 3:
        calibrated_y = 0.2126 * calibrated[..., 0] + 0.7152 * calibrated[..., 1] + 0.0722 * calibrated[..., 2]
    else:
        calibrated_y = calibrated

    return np.clip(calibrated_y, 0, 1)


def calculate_skew_angle(fiducials):
    """优化后的倾斜角度计算，仅使用水平线角度"""
    # 仅计算水平线角度 (使用左上和右上标记)
    dx = fiducials["URC"][0] - fiducials["ULC"][0]
    dy = fiducials["URC"][1] - fiducials["ULC"][1]
    return np.degrees(np.arctan2(dy, dx))  # 直接返回水平角度


def correct_perspective(image, fiducials, dpi=DEFAULT_DPI):
    """改进的透视校正函数，返回校正后的图像和实际使用的倾斜角度"""
    # 1. 计算原始倾斜角度
    skew_angle = calculate_skew_angle(fiducials)

    # 2. 旋转校正（仅当角度超过阈值时才进行）
    rotation_threshold = 0.01  # 可设为0.01度避免微小角度调整
    if abs(skew_angle) > rotation_threshold:
        # 记录实际应用的旋转角度（取反）
        applied_angle = -skew_angle
        image = correct_skew(image, applied_angle, dpi)
        fiducials = update_fiducials_after_rotation(fiducials, skew_angle, image.shape)
    else:
        applied_angle = 0.0  # 无显著倾斜时不旋转

    # 3. 转换为像素坐标
    def to_px(p):
        return [p[0] * dpi / MM_TO_INCH, p[1] * dpi / MM_TO_INCH]

    src = np.array([to_px(fiducials[k]) for k in ["ULC", "URC", "LRC", "LLC"]], np.float32)

    # 4. 计算目标区域的宽度和高度
    w = np.linalg.norm(src[1] - src[0])  # 上边长度
    h = np.linalg.norm(src[3] - src[0])  # 左边长度

    # 5. 定义目标四边形
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    # 6. 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(src, dst)
    corrected_image = cv2.warpPerspective(image, M, (int(w), int(h)))

    return corrected_image, applied_angle  # 返回校正后的图像和实际旋转角度


def update_fiducials_after_rotation(fiducials, angle, image_shape, dpi=DEFAULT_DPI):
    """旋转后更新基准点坐标的辅助函数"""
    h, w = image_shape[:2]
    center = (w / 2, h / 2)

    # 转换为像素坐标
    def to_px(p):
        return [p[0] * dpi / MM_TO_INCH, p[1] * dpi / MM_TO_INCH]

    # 转换为毫米坐标
    def to_mm(p):
        return [p[0] * MM_TO_INCH / dpi, p[1] * MM_TO_INCH / dpi]

    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    updated = {}
    for key in fiducials:
        # 转换为像素坐标
        px = to_px(fiducials[key])
        # 应用旋转
        rotated_px = np.dot(M, np.array([px[0], px[1], 1]))
        # 转换回毫米坐标
        updated[key] = to_mm(rotated_px)

    return updated


def correct_skew(image, angle, dpi):
    """校正图像倾斜"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新的边界尺寸
    radians = np.radians(angle)
    new_w = int(abs(h * np.sin(radians)) + abs(w * np.cos(radians)))
    new_h = int(abs(h * np.cos(radians)) + abs(w * np.sin(radians)))

    # 调整旋转矩阵
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # 执行旋转
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)


def calibrate_L_star(image, oecf_path=None, dpi=DEFAULT_DPI):
    """计算CIE L*值（整合了反射率校准）"""
    # 1. 获取校准后的反射率
    calibrated = calibrate_reflectance(image, oecf_path, dpi)

    # 2. 转换为L*值
    Y = np.clip(calibrated, 1e-5, 1.0)  # 避免零值
    L = np.where(Y > 0.008856,
                 116 * (Y ** (1 / 3)) - 16,
                 903.3 * Y)
    return L


def iso_banding_metric(y_profile, dpi):
    """根据ISO标准计算banding指标"""
    # 确保输入是一维数组
    y_profile = np.asarray(y_profile).flatten()

    # 1. 转换为L* (CIELAB亮度)
    Y = np.clip(y_profile, 1e-5, 1.0)  # 避免负值
    L = np.where(Y > 0.008856,
                 116 * (Y ** (1 / 3)) - 16,
                 903.3 * Y)

    # 2. 应用调制函数Q(f)
    n = len(L)
    spacing = MM_TO_INCH / dpi  # 像素间距(mm)
    freq = fftfreq(n, d=spacing)  # 频率单位: cycles/mm

    # 调制传递函数 (文档2-Step 9-2f)
    Q = 0.617 / (1 + 0.4 * (np.abs(freq) / 0.074) ** 1.33)
    Q[np.abs(freq) > 0.5] = 0  # 0.5 cycles/mm截止频率

    # 频域滤波
    L_fft = fft(L)
    L_filtered = ifft(L_fft * Q).real

    # 3. 三级高斯滤波 (文档2-Step 9-2g)
    sigmas = [50, 5, 0.5]  # mm
    D = []
    for sigma in sigmas:
        # 计算高斯核大小 (奇数)
        kernel_size = int(6 * sigma * dpi / MM_TO_INCH)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 应用高斯滤波 (确保是一维滤波)
        filtered = cv2.GaussianBlur(L_filtered.reshape(-1, 1), (kernel_size, 1), sigma * dpi / MM_TO_INCH)
        D.append(filtered.flatten())

    # 4. Tent-pole求和 (文档2-Step 9-2j)
    extrema = []
    for band in D:
        # 确保band是一维数组
        band = band.flatten()

        # 找到所有极值点 (排除边缘5mm)
        margin = int(5 * dpi / MM_TO_INCH)
        if margin * 2 >= len(band):
            margin = len(band) // 4  # 如果数组太小，减小margin

        peaks, _ = find_peaks(band[margin:-margin], height=np.mean(band))
        valleys, _ = find_peaks(-band[margin:-margin], height=-np.mean(band))

        # 收集极值
        extrema.extend(band[peaks + margin] - np.mean(band))
        extrema.extend(-(band[valleys + margin] - np.mean(band)))

    # 计算M值 (文档2-Step 9-2j)
    if len(extrema) == 0:
        return 0.0  # 没有检测到极值

    sorted_extrema = np.sort(np.abs(extrema))[-10:]  # 取最大的10个极值
    M = np.sqrt(np.sum(sorted_extrema ** 2))

    # 最终banding指标 (文档2-Step 9-2k)
    return 3.66 * M


def extract_roi(image, config, dpi):
    """改进的ROI提取函数，使用基准点坐标系"""
    # 获取ROI参数(mm)
    x_mm = config["ROI"]["X"]
    y_mm = config["ROI"]["Y"]
    w_mm = config["ROI"]["Width"]
    h_mm = config["ROI"]["Height"]

    # 转换为像素坐标 - 使用统一DPI（如需要不同DPI可修改此处）
    x_px, y_px, w_px, h_px = convert_bang_to_px(x_mm, y_mm, w_mm, h_mm, dpi_x=380, dpi_y=209)

    # 计算ROI边界
    x1 = max(0, x_px)
    y1 = max(0, y_px)
    x2 = min(x_px + w_px, image.shape[1] - 1)
    y2 = min(y_px + h_px, image.shape[0] - 1)

    return image[y1:y2, x1:x2]


def convert_bang_to_px(x_mm, y_mm, dx_mm, dy_mm, dpi_x=380, dpi_y=209):
    """毫米到像素的转换（支持不同X/Y方向DPI）"""
    x_px = int(x_mm * dpi_x / MM_TO_INCH)
    y_px = int(y_mm * dpi_y / MM_TO_INCH)
    dx_px = int(dx_mm * dpi_x / MM_TO_INCH)
    dy_px = int(dy_mm * dpi_y / MM_TO_INCH)
    return (x_px, y_px, dx_px, dy_px)


def measure_banding(image, config, dpi, direction='vertical'):
    """测量指定方向的banding"""
    # 提取ROI
    roi = extract_roi(image, config, dpi)

    # 根据方向计算剖面
    if direction == 'vertical':
        profile = np.mean(roi, axis=0)  # 垂直条带: x方向平均
    else:  # horizontal
        profile = np.mean(roi, axis=1)  # 水平条带: y方向平均

    # 计算banding指标
    return iso_banding_metric(profile, dpi)


def visualize_results(image, roi, vertical_score, horizontal_score, skew_angle):
    """可视化测量结果"""
    plt.figure(figsize=(15, 6))

    # 原始图像和ROI
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("校准后图像")
    plt.axis('off')

    # ROI区域
    plt.subplot(1, 3, 2)
    plt.imshow(roi, cmap='gray')
    plt.title(
        f"ROI区域\n尺寸: {roi.shape[1] / DEFAULT_DPI * MM_TO_INCH:.1f}x{roi.shape[0] / DEFAULT_DPI * MM_TO_INCH:.1f}mm")
    plt.axis('off')

    # 测量结果
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.6,
             f"垂直条带: {vertical_score:.6f}\n"
             f"水平条带: {horizontal_score:.6f}\n"
             f"倾斜角度: {skew_angle:.2f}°",
             fontsize=12)
    plt.axis('off')
    plt.title("测量结果")

    plt.tight_layout()
    plt.show()


def save_results(results, output_path=None):
    """按文档1格式保存结果（修复float32序列化问题）"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"banding_results_{timestamp}"

    # 转换所有NumPy类型为Python原生类型
    def convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        return o

    # 文本格式输出
    with open(f"{output_path}.txt", "w") as f:
        f.write("Image file name\tROI:H(mm)\tROI:V(mm)\t"
                "Vertical L*\tVertical score\t"
                "Horizontal L*\tHorizontal score\tSkew_angle(deg.)\n")
        f.write(f"{results['image_path']}\t"
                f"{results['roi_height_mm']:.2f}\t"
                f"{results['roi_width_mm']:.2f}\t"
                f"{results['vertical_L_star']:.3f}\t"
                f"{results['vertical_score']:.5f}\t"
                f"{results['horizontal_L_star']:.3f}\t"
                f"{results['horizontal_score']:.5f}\t"
                f"{results['skew_angle']:.6f}\n")

    # JSON格式输出（添加类型转换）
    with open(f"{output_path}.json", "w") as f:
        json.dump({k: convert(v) for k, v in results.items()}, f, indent=4)


def measure_banding_attributes(image_path, config_path=None, oecf_path=None, dpi=DEFAULT_DPI, debug=False):
    """主测量函数（整合新逻辑）"""
    # 1. 加载原始图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 2. 加载配置文件
    with open(config_path, 'r') as f:
        config_content = f.read()
    config = parse_config_file(config_content)

    # 3. 提取ROI信息
    if 'Band' not in config['Measurement']:
        raise ValueError("配置文件中缺少Band测量区域信息")

    band_info = config['Measurement']['Band']
    roi_config = {
        'X': band_info['X'] - band_info['dX'] / 2,
        'Y': band_info['Y'] - band_info['dY'] / 2,
        'Width': band_info['dX'],
        'Height': band_info['dY']
    }

    # 4. 应用OECF校准
    calibrated_reflectance = calibrate_reflectance(img, oecf_path)

    # 5. 几何校正（先旋转再透视），同时获取实际应用的旋转角度
    corrected_img, applied_skew_angle = correct_perspective(calibrated_reflectance, config["Fiducials"], dpi)

    # 6. 计算L*值
    L_star = np.where(corrected_img > 0.008856,
                      116 * np.power(corrected_img, 1 / 3) - 16,
                      903.3 * corrected_img)

    # 7. 提取ROI区域
    roi = extract_roi(L_star, {'ROI': roi_config}, dpi)

    # 8. 测量banding
    vertical_profile = np.mean(roi, axis=0)
    horizontal_profile = np.mean(roi, axis=1)
    vertical_score = iso_banding_metric(vertical_profile, dpi)
    horizontal_score = iso_banding_metric(horizontal_profile, dpi)

    # 构建结果字典
    results = {
        "image_path": image_path,
        "oecf_applied": oecf_path is not None,
        "config_used": config_path is not None,
        "roi_width_mm": roi_config['Width'],
        "roi_height_mm": roi_config['Height'],
        "vertical_L_star": np.mean(vertical_profile),
        "vertical_score": vertical_score,
        "horizontal_L_star": np.mean(horizontal_profile),
        "horizontal_score": horizontal_score,
        "skew_angle": round(applied_skew_angle, 6),
        "dpi": dpi,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 打印结果
    print("\n测量结果:")
    print(f"{'参数':<20} | {'测量值':<15}")
    print("-" * 50)
    print(f"{'ROI宽度(mm)':<20} | {results['roi_width_mm']:<15.2f}")
    print(f"{'ROI高度(mm)':<20} | {results['roi_height_mm']:<15.2f}")
    print(f"{'垂直L*':<20} | {results['vertical_L_star']:<15.3f}")
    print(f"{'垂直分数':<20} | {results['vertical_score']:<15.5f}")
    print(f"{'水平L*':<20} | {results['horizontal_L_star']:<15.3f}")
    print(f"{'水平分数':<20} | {results['horizontal_score']:<15.5f}")
    print(f"{'倾斜角度(°)':<20} | {applied_skew_angle:<15.6f}")

    if debug:
        visualize_results(L_star, roi, vertical_score, horizontal_score, applied_skew_angle)

    return results


if __name__ == "__main__":
    # 示例用法
    image_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000001_600_Image1.tif"
    config_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/data/Config_Conformance_Band2.txt"  # 可选配置文件
    oecf_path = "D:/PycharmProjects/pythonProject/image2-1-oecf.xlsx"  # 可选OECF文件

    # 执行测量
    results = measure_banding_attributes(
        image_path=image_path,
        config_path=config_path,
        oecf_path=oecf_path,
        dpi=600,  # 必须使用600ppi
        debug=True
    )

    # 保存结果
    save_results(results)