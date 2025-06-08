import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks
import json
from datetime import datetime
import pandas as pd
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


def px_to_mm(px, dpi=DEFAULT_DPI):
    """像素到毫米的转换"""
    return px * MM_TO_INCH / dpi


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


def calibrate_reflectance(image, oecf_path=None, dpi=DEFAULT_DPI, measurement_type="banding"):
    """改进的反射率校准函数，根据测量类型选择正确的OECF应用方式

    Args:
        measurement_type:
            "banding" - 使用sRGB转换（用于banding/mottle/graininess）
            "line" - 使用Rvis转换（用于字符和线条测量）
    """
    # 1. 加载OECF数据
    oecf_data = load_oecf_data_from_excel(oecf_path)
    verify_oecf(oecf_data)

    # 2. 图像预处理
    if image.dtype == np.uint16:
        img_float = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # 3. 应用OECF校准（根据测量类型选择转换方式）
    calibrated = np.zeros_like(img_float)
    for c in range(3):
        # 创建查找表
        lut = np.interp(np.linspace(0, 1, 256),
                        np.linspace(0, 1, len(oecf_data[c])),
                        oecf_data[c])

        # 应用LUT
        channel = np.clip(img_float[..., c] * 255, 0, 255).astype(np.uint8)
        calibrated[..., c] = lut[channel]

    # 4. 转换为反射率（关键修改点）
    if measurement_type == "banding":
        # 使用sRGB转换公式 (文档2附录2)
        calibrated_y = 0.2126 * calibrated[..., 0] + 0.7152 * calibrated[..., 1] + 0.0722 * calibrated[..., 2]
    else:
        # 使用Rvis转换 (文档1第5.2.5节)
        calibrated_y = 0.93137 * calibrated[..., 0] + 0.933 * calibrated[..., 1] + 0.933 * calibrated[..., 2]

    return np.clip(calibrated_y, 0, 1)


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
    """根据ISO 24790标准计算banding指标"""
    L = np.asarray(y_profile).flatten()  # 使用输入参数y_profile（即L*剖面）

    # 1. 应用调制函数Q(f)
    n = len(L)
    spacing = MM_TO_INCH / dpi  # 像素间距(mm)
    freq = fftfreq(n, d=spacing)  # cycles/mm

    # 正确的调制传递函数
    Q = np.zeros_like(freq)
    mask = (np.abs(freq) <= 0.5) & (np.abs(freq) > 0)  # 0.5 cycles/mm截止频率
    f = np.abs(freq[mask])
    Q[mask] = 0.617 + 0.40 * np.arctan(1.33 * np.log(f / 0.074))

    # 2. 频域滤波
    L_fft = fft(L - np.mean(L))  # 去除DC分量
    L_filtered = ifft(L_fft * Q).real

    # 3. 三级高斯滤波（连续处理）
    from scipy.ndimage import gaussian_filter1d
    D1 = gaussian_filter1d(L_filtered, sigma=50/25.4*dpi)
    D2 = gaussian_filter1d(L_filtered, sigma=5/25.4*dpi) - D1
    D3 = gaussian_filter1d(L_filtered, sigma=0.5/25.4*dpi) - gaussian_filter1d(L_filtered, sigma=5/25.4*dpi)

    # 4. Tent-pole求和
    extrema = []
    for band in [D1, D2, D3]:  # 对三个频带分别处理
        band = band.flatten()
        margin = max(10, int(5 * dpi / MM_TO_INCH))  # 5mm边缘

        # 找极值点
        peaks, _ = find_peaks(band[margin:-margin])
        valleys, _ = find_peaks(-band[margin:-margin])

        extrema.extend(band[peaks + margin])
        extrema.extend(-band[valleys + margin])

    # 按绝对值降序排序
    sorted_extrema = np.sort(np.abs(extrema))[::-1]

    # Tent-pole求和 - 限制极值点数量最多为100个
    p = 2  # 衰减因子
    M = 0.0
    max_extrema = min(100, len(sorted_extrema))  # 限制最多使用100个极值点
    for i in range(max_extrema):
        delta = sorted_extrema[i]
        M += delta / (p ** i)

    # 5. 最终banding指标
    return 3.66 * np.sqrt(M)


def extract_roi(image, roi_rect, dpi):
    """改进的ROI提取函数"""
    # 解包ROI坐标
    x1, y1, x2, y2 = roi_rect

    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1] - 1, x2)
    y2 = min(image.shape[0] - 1, y2)

    return image[y1:y2, x1:x2]


def adjust_reference_point(image, point_name, initial_position):
    """改进的基准点调整函数，使用十字光标"""
    # 转换为灰度图像用于显示
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        display_img = image.copy()

    # 归一化到0-255范围
    if display_img.dtype != np.uint8:
        display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 创建彩色显示图像
    img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # 创建放大区域
    zoom_factor = 4  # 放大倍数
    zoom_size = 100  # 放大区域大小(像素)

    current_pos = list(initial_position)
    step = 1  # 移动步长(像素)

    while True:
        # 创建新的显示图像
        img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        # 绘制十字光标(红色)
        cv2.line(img_copy, (current_pos[0] - 20, current_pos[1]), (current_pos[0] + 20, current_pos[1]),
                 (0, 0, 255), 2)
        cv2.line(img_copy, (current_pos[0], current_pos[1] - 20), (current_pos[0], current_pos[1] + 20),
                 (0, 0, 255), 2)

        # 绘制基准点名称
        cv2.putText(img_copy, point_name, (current_pos[0] + 25, current_pos[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 创建放大视图
        x1 = max(0, current_pos[0] - zoom_size // 2)
        y1 = max(0, current_pos[1] - zoom_size // 2)
        x2 = min(img_copy.shape[1], current_pos[0] + zoom_size // 2)
        y2 = min(img_copy.shape[0], current_pos[1] + zoom_size // 2)

        zoomed = img_copy[y1:y2, x1:x2]
        if zoomed.size > 0:
            zoomed = cv2.resize(zoomed, None, fx=zoom_factor, fy=zoom_factor,
                                interpolation=cv2.INTER_NEAREST)

            # 在放大视图中绘制更精细的十字线
            center_x, center_y = zoomed.shape[1] // 2, zoomed.shape[0] // 2
            cv2.line(zoomed, (center_x - 30, center_y), (center_x + 30, center_y),
                     (0, 255, 0), 1)
            cv2.line(zoomed, (center_x, center_y - 30), (center_x, center_y + 30),
                     (0, 255, 0), 1)

            # 将放大视图放置在主图像旁边
            if img_copy.shape[1] > zoomed.shape[1] + 20:
                img_copy[10:10 + zoomed.shape[0],
                img_copy.shape[1] - zoomed.shape[1] - 10:img_copy.shape[1] - 10] = zoomed
                cv2.rectangle(img_copy,
                              (img_copy.shape[1] - zoomed.shape[1] - 10, 10),
                              (img_copy.shape[1] - 10, 10 + zoomed.shape[0]),
                              (0, 255, 0), 1)

        # 添加操作说明
        cv2.putText(img_copy, f"Adjust {point_name} point", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_copy, "Use arrow keys to move, ENTER to confirm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_copy, "SPACE to toggle step size (1/10 px)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_copy, f"Current position: {current_pos}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_copy, f"Step size: {step} px", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示图像
        cv2.namedWindow(f"Adjust {point_name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Adjust {point_name}", 800, 600)
        cv2.imshow(f"Adjust {point_name}", img_copy)

        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # ENTER键
            break
        elif key == 27:  # ESC键
            cv2.destroyAllWindows()
            return None
        elif key == ord(' '):  # 空格键
            step = 10 if step == 1 else 1
        elif key == ord('w'):  # 使用WASD代替方向键
            current_pos[1] = max(0, current_pos[1] - step)
        elif key == ord('s'):
            current_pos[1] = min(image.shape[0] - 1, current_pos[1] + step)
        elif key == ord('a'):
            current_pos[0] = max(0, current_pos[0] - step)
        elif key == ord('d'):
            current_pos[0] = min(image.shape[1] - 1, current_pos[0] + step)

    cv2.destroyAllWindows()
    return tuple(current_pos)


def calculate_skew_angle(image, dpi):
    """计算图像倾斜角度(通过四个基准点)"""
    # 转换为灰度图像用于显示
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        display_img = image.copy()

    # 归一化到0-255范围
    if display_img.dtype != np.uint8:
        display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 定义初始基准点位置
    initial_points = {
        'ULC': (114, 90),  # 左上角
        'URC': (4051, 100),  # 右上角
        'LRC': (4049, 2855),  # 右下角
        'LLC': (114, 2840)  # 左下角
    }

    # 让用户调整四个基准点
    adjusted_points = {}
    for point_name, initial_pos in initial_points.items():
        print(f"请调整 {point_name} 基准点位置...")
        adjusted_pos = adjust_reference_point(display_img, point_name, initial_pos)
        if adjusted_pos is None:
            print("用户取消了基准点调整")
            return 0.0  # 返回0角度表示无倾斜
        adjusted_points[point_name] = adjusted_pos
        print(f"{point_name} 已设置为: {adjusted_pos}")

    # 计算倾斜角度(使用左上和右上点)
    ulc = adjusted_points["ULC"]
    urc = adjusted_points["URC"]

    # 计算水平线的角度(以度为单位)
    delta_y = urc[1] - ulc[1]
    delta_x = urc[0] - ulc[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    # 实际旋转角度 = -原始角度(取反)
    actual_angle = -angle_deg

    # 可视化基准点和计算的角度
    vis_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    for point_name, pos in adjusted_points.items():
        # 绘制十字光标
        cv2.line(vis_img, (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), (0, 0, 255), 2)
        cv2.line(vis_img, (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), (0, 0, 255), 2)
        cv2.putText(vis_img, point_name, (pos[0] + 10, pos[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 绘制基准线
    cv2.line(vis_img, ulc, urc, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Skew angle: {angle_deg:.2f}°", (vis_img.shape[1] // 2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Actual rotation: {actual_angle:.2f}°", (vis_img.shape[1] // 2 - 100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.namedWindow("Skew Angle Calculation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skew Angle Calculation", 800, 600)
    cv2.imshow("Skew Angle Calculation", vis_img)
    cv2.waitKey(3000)  # 显示3秒
    cv2.destroyAllWindows()

    return actual_angle


def rotate_image(image, angle):
    """旋转图像(保持完整图像)
    Args:
        image: 输入图像
        angle: 旋转角度（正值为逆时针，负值为顺时针）
    """
    if angle == 0:
        return image.copy()

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心
    center = (width // 2, height // 2)

    # 获取旋转矩阵（直接使用传入的角度，因为calculate_skew_angle已经处理了方向）
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新的边界尺寸
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2

    # 执行旋转
    if len(image.shape) == 3:
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return rotated_image

def visualize_results(image, roi, roi_rect, overall_score, dpi):
    """可视化测量结果"""
    # 计算ROI尺寸(mm)
    roi_width_px = roi_rect[2] - roi_rect[0]
    roi_height_px = roi_rect[3] - roi_rect[1]
    roi_width_mm = px_to_mm(roi_width_px, dpi)
    roi_height_mm = px_to_mm(roi_height_px, dpi)

    plt.figure(figsize=(15, 6))

    # 原始图像和ROI
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("校准后图像")
    plt.axis('off')

    # ROI区域
    plt.subplot(1, 3, 2)
    plt.imshow(roi, cmap='gray')
    plt.title(f"ROI区域\n尺寸: {roi_width_mm:.1f}x{roi_height_mm:.1f}mm")
    plt.axis('off')

    # 测量结果
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.6,
             f"ROI宽度: {roi_width_mm:.2f}mm\n"
             f"ROI高度: {roi_height_mm:.2f}mm\n"
             f"整体条带分数: {overall_score:.6f}",
             fontsize=12)
    plt.axis('off')
    plt.title("测量结果")

    plt.tight_layout()
    plt.show()


def save_results(results, output_path=None):
    """保存结果"""
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
                "Average L*\tOverall score\t"
                "Skew angle\n")
        f.write(f"{results['image_path']}\t"
                f"{results['roi_height_mm']:.2f}\t"
                f"{results['roi_width_mm']:.2f}\t"
                f"{results['average_L_star']:.3f}\t"
                f"{results['overall_score']:.5f}\t"
                f"{results['skew_angle']:.2f}\n")

    # JSON格式输出（添加类型转换）
    with open(f"{output_path}.json", "w") as f:
        json.dump({k: convert(v) for k, v in results.items()}, f, indent=4)


def select_roi_interactive(image, dpi):
    """完全手动框选ROI区域（优化可视化版本）"""
    # 转换为灰度图像用于显示
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        display_img = image.copy()

    # 归一化到0-255范围
    if display_img.dtype != np.uint8:
        display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 创建窗口
    cv2.namedWindow("Select ROI (Drag & Release)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI (Drag & Release)", 800, 600)

    # 创建副本用于绘制
    img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # 添加操作指引文本
    cv2.putText(img_copy, "Drag mouse to select ROI", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_copy, "Press ENTER to confirm, ESC to cancel", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ROI选择状态变量
    roi_selected = False
    roi_rect = (0, 0, 0, 0)
    dragging = False
    start_point = (-1, -1)
    current_point = (-1, -1)

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_selected, roi_rect, dragging, start_point, current_point, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start_point = (x, y)
            current_point = (x, y)
            # 重置选择
            roi_selected = False
            img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            # 重新添加指引文本
            cv2.putText(img_copy, "Drag mouse to select ROI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img_copy, "Press ENTER to confirm, ESC to cancel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                current_point = (x, y)
                # 绘制当前选择框
                img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                # 添加指引文本
                cv2.putText(img_copy, "Drag mouse to select ROI", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img_copy, "Press ENTER to confirm, ESC to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 绘制矩形
                cv2.rectangle(img_copy, start_point, current_point, (0, 255, 0), 2)

                # 计算并显示实时尺寸（像素和毫米）
                width_px = abs(current_point[0] - start_point[0])
                height_px = abs(current_point[1] - start_point[1])
                width_mm = px_to_mm(width_px, dpi)
                height_mm = px_to_mm(height_px, dpi)

                # 在矩形下方显示尺寸信息
                text_pos = (min(start_point[0], current_point[0]),
                            max(start_point[1], current_point[1]) + 20)
                cv2.putText(img_copy,
                            f"{width_px}x{height_px} px ({width_mm:.1f}x{height_mm:.1f} mm)",
                            text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("Select ROI (Drag & Release)", img_copy)

        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            end_point = (x, y)
            current_point = end_point
            # 确保是有效的矩形
            min_roi_size = max(10, int(5 * dpi / MM_TO_INCH))  # 至少5mm或10px

            if (abs(start_point[0] - end_point[0]) > min_roi_size and
                    abs(start_point[1] - end_point[1]) > min_roi_size):
                roi_rect = (
                    min(start_point[0], end_point[0]),
                    min(start_point[1], end_point[1]),
                    max(start_point[0], end_point[0]),
                    max(start_point[1], end_point[1])
                )
                roi_selected = True
                # 绘制最终选择框
                img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img_copy, (roi_rect[0], roi_rect[1]),
                              (roi_rect[2], roi_rect[3]), (0, 255, 0), 2)

                # 添加确认文本
                cv2.putText(img_copy, "ROI Selected! Press ENTER to confirm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_copy, "or drag to reselect, ESC to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示最终尺寸
                width_px = roi_rect[2] - roi_rect[0]
                height_px = roi_rect[3] - roi_rect[1]
                width_mm = px_to_mm(width_px, dpi)
                height_mm = px_to_mm(height_px, dpi)
                text_pos = (roi_rect[0], roi_rect[3] + 20)
                cv2.putText(img_copy,
                            f"Final: {width_px}x{height_px} px ({width_mm:.1f}x{height_mm:.1f} mm)",
                            text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("Select ROI (Drag & Release)", img_copy)
            else:
                print(f"请选择更大的区域（至少{min_roi_size}像素）")
                # 重置选择
                roi_selected = False
                img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                # 重新添加指引文本
                cv2.putText(img_copy, "Drag mouse to select ROI", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img_copy, "Press ENTER to confirm, ESC to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Select ROI (Drag & Release)", img_copy)

    # 初始显示
    cv2.setMouseCallback("Select ROI (Drag & Release)", mouse_callback)
    cv2.imshow("Select ROI (Drag & Release)", img_copy)

    # 等待用户选择
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and roi_selected:  # Enter键确认
            break
        elif key == 27:  # ESC键退出
            cv2.destroyAllWindows()
            return None
        elif key == ord('r') and roi_selected:  # 按R键重新选择
            roi_selected = False
            img_copy = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            # 重新添加指引文本
            cv2.putText(img_copy, "Drag mouse to select ROI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img_copy, "Press ENTER to confirm, ESC to cancel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Select ROI (Drag & Release)", img_copy)

    cv2.destroyAllWindows()

    return roi_rect


def measure_banding_attributes(image_path, oecf_path=None, dpi=DEFAULT_DPI, debug=False):
    """主测量函数（整合新逻辑）"""
    # 1. 加载原始图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 2. 计算倾斜角度（在原始图像上计算）
    skew_angle = calculate_skew_angle(img, dpi)

    # 3. 应用OECF校准（在原始图像上）
    calibrated_reflectance = calibrate_reflectance(img, oecf_path, dpi, measurement_type="banding")

    # 4. 旋转校准后的图像
    if abs(skew_angle) > 0:
        calibrated_rotated = rotate_image(calibrated_reflectance, skew_angle)
        if debug:
            print(f"已旋转图像 {skew_angle:.2f} 度进行校正")
    else:
        calibrated_rotated = calibrated_reflectance

    # 5. 计算L*值（在旋转后的图像上）
    L_star = np.where(calibrated_rotated > 0.008856,
                      116 * np.power(calibrated_rotated, 1 / 3) - 16,
                      903.3 * calibrated_rotated)

    # 6. 手动框选ROI区域（在L*图像上）
    roi_rect = select_roi_interactive(L_star, dpi)
    if roi_rect is None:
        print("用户取消了ROI选择")
        return None

    # 7. 提取ROI区域的L*值
    roi_L = extract_roi(L_star, roi_rect, dpi)

    # 8. 计算整体banding指标（将ROI展平为1D数组）
    flattened_L = roi_L.flatten()
    overall_score = iso_banding_metric(flattened_L, dpi)

    # 计算ROI尺寸(mm)
    roi_width_px = roi_rect[2] - roi_rect[0]
    roi_height_px = roi_rect[3] - roi_rect[1]
    roi_width_mm = px_to_mm(roi_width_px, dpi)
    roi_height_mm = px_to_mm(roi_height_px, dpi)

    # 构建结果字典
    results = {
        "image_path": image_path,
        "oecf_applied": oecf_path is not None,
        "roi_width_mm": roi_width_mm,
        "roi_height_mm": roi_height_mm,
        "average_L_star": np.mean(flattened_L),
        "overall_score": overall_score,
        "skew_angle": skew_angle,
        "dpi": dpi,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 打印结果
    print("\n测量结果:")
    print(f"{'参数':<20} | {'测量值':<15}")
    print("-" * 50)
    print(f"{'ROI宽度(mm)':<20} | {results['roi_width_mm']:<15.2f}")
    print(f"{'ROI高度(mm)':<20} | {results['roi_height_mm']:<15.2f}")
    print(f"{'平均L*':<20} | {results['average_L_star']:<15.3f}")
    print(f"{'整体分数':<20} | {results['overall_score']:<15.5f}")
    print(f"{'倾斜角度(度)':<20} | {results['skew_angle']:<15.3f}")
    if debug:
        visualize_results(L_star, roi_L, roi_rect, overall_score, dpi)


    return results

if __name__ == "__main__":
    # 示例用法
    image_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000001_600_Image1.tif"
    oecf_path = "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/OECFfile/image2_2.xlsx"  # 可选OECF文件

    # 执行测量
    results = measure_banding_attributes(
        image_path=image_path,
        oecf_path=oecf_path,
        dpi=600,  # 必须使用600ppi
        debug=True
    )

    # 保存结果
    if results is not None:
        save_results(results)