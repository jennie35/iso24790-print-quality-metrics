from PIL import Image  # 引入 PIL 库以支持 DPI 设置
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import find_peaks

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def validate_peaks(peaks, hist):
    """验证直方图中的双峰是否有效"""
    if len(peaks) < 2:
        return False
    # 检查两个峰值的间距和幅度
    peak1, peak2 = peaks[0], peaks[1]
    if abs(peak1 - peak2) < 50:  # 峰值间距过小
        return False
    if hist[peak1] < 50 or hist[peak2] < 50:  # 峰值幅度过小
        return False
    return True


def detect_line_orientation(image):
    """
    检测图像中线条的方向（水平或竖直）
    :param image: 输入图像（灰度图）
    :return: 方向（'horizontal' 或 'vertical'），旋转后的图像
    """
    # 边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        raise ValueError("未检测到线条")

    # 计算所有直线的平均角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # 计算平均角度
    avg_angle = np.mean(angles)

    # 判断方向
    if abs(avg_angle) < 45:  # 接近水平方向
        orientation = 'horizontal'
        # 旋转图像 90 度
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:  # 接近竖直方向
        orientation = 'vertical'
        rotated_image = image

    return orientation, rotated_image


def find_line_center(image):
    """
    找到图像中线条的中心位置
    :param image: 输入图像（灰度图）
    :return: 线条中心坐标 (x, y)
    """
    # 二值化图像
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到线条")

    # 取最大轮廓（假设为待测线条）
    main_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的中心
    M = cv2.moments(main_contour)
    if M["m00"] == 0:
        raise ValueError("无法计算线条中心")
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy


def measure_line_width(binary_image, num_samples=10):
    """
    使用公式（19）计算线宽
    :param binary_image: 二值化图像（线条为白色，背景为黑色）
    :param num_samples: 采样点的数量
    :return: 平均线宽（像素）
    """
    height, width = binary_image.shape
    line_widths = []

    # 沿线条方向等间隔采样
    for i in range(num_samples):
        # 计算采样点的 y 坐标
        y = int(height * (i + 1) / (num_samples + 1))

        # 提取水平方向的像素剖面
        profile = binary_image[y, :]

        # 找到左边缘和右边缘的阈值位置
        left_edge = np.argmax(profile > 0)  # 第一个非零像素
        right_edge = len(profile) - np.argmax(profile[::-1] > 0) - 1  # 最后一个非零像素

        # 计算线宽
        line_width = right_edge - left_edge
        line_widths.append(line_width)

    # 计算平均线宽
    average_line_width = np.mean(line_widths)
    return average_line_width


def measure_roi_attributes(image_path, dpi=1200, debug=False):
    """
    测量ROI图像中的反射率阈值和线宽（符合ISO/IEC 24790-2017附录D）
    :param image_path: 输入图像路径
    :param dpi: 图像分辨率（每英寸点数，默认1200 DPI）
    :param debug: 是否显示中间处理结果
    :return: 阈值字典、线宽（毫米）
    """
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 检测线条方向并旋转图像
    orientation, rotated_gray = detect_line_orientation(gray)
    print(f"检测到线条方向: {orientation}")

    # 3. 找到线条中心
    cx, cy = find_line_center(rotated_gray)
    print(f"线条中心坐标: ({cx}, {cy})")

    # 4. 设置ROI的宽度和高度（单位：毫米）
    roi_width_mm = 2
    roi_height_mm = 10

    # 将ROI尺寸从毫米转换为像素
    roi_width_px = int((roi_width_mm / 25.4) * dpi)
    roi_height_px = int((roi_height_mm / 25.4) * dpi)

    # 以线条中心为中心裁剪ROI
    x1 = max(cx - roi_width_px // 2, 0)
    y1 = max(cy - roi_height_px // 2, 0)
    x2 = min(x1 + roi_width_px, rotated_gray.shape[1])
    y2 = min(y1 + roi_height_px, rotated_gray.shape[0])
    roi = rotated_gray[y1:y2, x1:x2]

    # 5. 自动检测背景和线条区域（双峰法）
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    peaks, _ = find_peaks(hist.flatten(), height=50, distance=50)
    if not validate_peaks(peaks, hist):
        raise ValueError("反射率分布不呈现有效的双峰形状")

    # 6. 将灰度值归一化到[0, 1]范围（反射率）
    r_min = peaks[0] / 255.0
    r_max = peaks[1] / 255.0

    # 7. 计算阈值（R10, R40, R70, R75）
    r10 = r_min + 0.1 * (r_max - r_min)
    r40 = r_min + 0.4 * (r_max - r_min)
    r70 = r_min + 0.7 * (r_max - r_min)
    r75 = r_min + 0.75 * (r_max - r_min)

    # 8. 二值化并提取线条轮廓
    _, binary = cv2.threshold(roi, int(r40 * 255), 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到线条")

    # 取最大轮廓（假设为待测线条）
    main_contour = max(contours, key=cv2.contourArea)

    # 9. 计算线条的最小外接矩形
    rect = cv2.minAreaRect(main_contour)
    (center, (width, height), angle) = rect

    # 确定线条方向
    if width > height:
        line_angle = angle
        line_length = width
        line_width = height
    else:
        line_angle = angle + 90
        line_length = height
        line_width = width

    # 10. 使用公式（19）计算线宽
    line_width_px = measure_line_width(binary, num_samples=10)

    # 11. 将线宽从像素转换为毫米
    line_width_mm = (line_width_px / dpi) * 25.4

    # 12. 计算线条图像密度
    r25 = r_min + 0.25 * (r_max - r_min)
    line_density = -np.log10(r25)

    # 13. 可视化结果
    if debug:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].imshow(roi, cmap='gray')
        ax[0, 0].set_title("原始ROI图像")
        ax[0, 0].axis('off')

        ax[0, 1].plot(hist)
        ax[0, 1].scatter(peaks, hist[peaks], color='red', marker='x')
        ax[0, 1].set_title("反射率直方图与峰值")
        ax[0, 1].set_xlabel("反射率值")
        ax[0, 1].set_ylabel("频数")

        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, [main_contour], -1, (0, 255, 0), 1)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(overlay, [box], 0, (0, 0, 255), 2)
        ax[1, 0].imshow(overlay)
        ax[1, 0].set_title("边缘检测与线宽测量")
        ax[1, 0].axis('off')

        ax[1, 1].axis('off')
        text = f"""
        R_max (背景): {r_max:.2f}
        R_min (线条): {r_min:.2f}
        R10 (内边界): {r10:.2f}
        R40 (边缘阈值): {r40:.2f}
        R70 (外边界): {r70:.2f}
        R75: {r75:.2f}
        线宽: {line_width_mm:.2f} mm
        线条图像密度: {line_density:.2f}
        """
        ax[1, 1].text(0.1, 0.5, text, fontsize=12)
        plt.tight_layout()
        plt.show()

    return {
        "R_max": r_max,
        "R_min": r_min,
        "R10": r10,
        "R40": r40,
        "R70": r70,
        "R75": r75,
        "LineWidth_mm": line_width_mm,
        "LineDensity": line_density
    }


def create_bitmap_from_table(table, dpi=800):
    """
    根据附录 A 中的位图数据生成图像，并设置 DPI
    :param table: 位图数据（字符串列表）
    :param dpi: 分辨率（默认 800 DPI）
    :return: 生成的图像（PIL 图像）
    """
    # 将位图数据转换为二维数组
    bitmap = np.array([[int(char) for char in row] for row in table], dtype=np.uint8)

    # 将 0 和 1 转换为 0（黑色）和 255（白色）
    bitmap = np.where(bitmap == 1, 255, 0)

    # 将 NumPy 数组转换为 PIL 图像
    bitmap_pil = Image.fromarray(bitmap, mode='L')  # 使用灰度模式 'L'

    # 设置 DPI
    bitmap_pil.info['dpi'] = (dpi, dpi)

    return bitmap_pil


def apply_bitmap_to_image(image_path, bitmap, output_path, dpi=800):
    """
    将位图应用到测试图像上，并设置 DPI
    :param image_path: 测试图像路径
    :param bitmap: 位图图像（PIL 图像）
    :param output_path: 输出图像路径
    :param dpi: 分辨率（默认 800 DPI）
    """
    # 读取测试图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法读取测试图像")

    # 将位图调整到与测试图像相同的大小
    bitmap_resized = bitmap.resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS)

    # 将位图与测试图像结合
    combined_image = cv2.addWeighted(img, 0.7, np.array(bitmap_resized), 0.3, 0)

    # 保存输出图像，并设置 DPI
    combined_image_pil = Image.fromarray(combined_image, mode='L')  # 使用灰度模式 'L'
    combined_image_pil.save(output_path, dpi=(dpi, dpi))


# 附录 A 中的位图数据
bitmap_tables = {
    "Line 1": [
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
        "000000000011110000000000",
    ],
    "Line 2": [
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
        "000000001111111100000000",
    ],
    # 其他位图数据...
}

# 生成并保存位图图像
for line_name, table in bitmap_tables.items():
    bitmap = create_bitmap_from_table(table, dpi=800)
    bitmap.save(f"{line_name}.png")  # 保存为 PNG 格式，并设置 DPI
    print(f"已生成 {line_name}.png (800 DPI)")

# 示例调用
if __name__ == "__main__":
    # 测试图像路径
    test_image_path = "D:/PycharmProjects/pythonProject/ISO24790/output_rois23/L13_Mark.tif"
    dpi = 1200  # 图像分辨率（DPI）

    # 将位图应用到测试图像上
    for line_name in bitmap_tables.keys():
        bitmap_path = f"{line_name}.png"
        output_path = f"{line_name}_applied.png"

        # 读取位图
        bitmap = Image.open(bitmap_path)
        if bitmap is None:
            print(f"无法读取位图: {bitmap_path}")
            continue

        # 将位图应用到测试图像上
        apply_bitmap_to_image(test_image_path, bitmap, output_path, dpi=800)
        print(f"已将 {line_name} 应用到测试图像，保存为 {output_path} (800 DPI)")

        # 测量整合后的图像
        try:
            results = measure_roi_attributes(output_path, dpi=dpi, debug=True)
            print(f"{line_name} 的测量结果:")
            for key, value in results.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"{line_name} 错误: {e}")