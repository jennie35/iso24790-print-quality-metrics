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
    检测图像中线条的方向并确定需要的狭缝方向
    :param image: 输入图像（灰度图）
    :return: (方向, 旋转后图像, 需要的狭缝方向)
    """
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        raise ValueError("未检测到线条")

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    avg_angle = np.mean(angles)

    if abs(avg_angle) < 45:  # 水平线条
        orientation = 'horizontal'
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        slit_direction = 'vertical'  # 水平线条需要竖直狭缝
    else:  # 垂直线条
        orientation = 'vertical'
        rotated_image = image
        slit_direction = 'horizontal'  # 垂直线条需要水平狭缝

    return orientation, rotated_image, slit_direction


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



def detect_line_orientation(image):
    """
    检测图像中线条的方向
    :param image: 输入图像（灰度图）
    :return: (方向, 旋转后图像)
    """
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        raise ValueError("未检测到线条")

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    avg_angle = np.mean(angles)

    if abs(avg_angle) < 45:  # 水平线条
        orientation = 'horizontal'
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:  # 垂直线条
        orientation = 'vertical'
        rotated_image = image

    return orientation, rotated_image


def calculate_line_width_slit(image):
    """
    严格遵循ISO/IEC 24790附录D的狭缝孔径平均法计算线宽
    固定使用竖直狭缝(vertical slit)测量垂直线条（根据Figure D.6）

    :param image: 输入图像(BGR格式)
    :return: (线宽mm, 标记图像, 反射率曲线, R40阈值, 扫描起始点, 扫描结束点)
    """
    # 1. 预处理图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 2. 固定设置竖直狭缝参数（根据Figure D.6）
    slit_width_px = 3  # 狭缝宽度（像素）
    scan_step = 1  # 扫描步长（像素）

    # 3. 确定扫描区域（图像中央水平区域）
    x_center = width // 2
    scan_start = max(0, x_center - 100)  # 从中心向左100像素开始
    scan_end = min(width, x_center + 100)  # 到中心向右100像素结束

    # 4. 竖直狭缝扫描过程（沿x轴水平移动）
    reflectance_profiles = []
    for x_pos in range(scan_start, scan_end, scan_step):
        # 提取竖直狭缝区域（垂直条带）
        slit = gray[:, x_pos - slit_width_px // 2: x_pos + slit_width_px // 2]
        # 计算狭缝平均反射率（标准化到0-1）
        avg_reflectance = np.mean(slit) / 255.0
        reflectance_profiles.append(avg_reflectance)

    # 5. 确定Rmax和Rmin（背景和线条的反射率极值）
    R_max = np.max(reflectance_profiles)
    R_min = np.min(reflectance_profiles)

    # 6. 计算R40阈值（标准3.12）
    R40 = R_min + 0.4 * (R_max - R_min)

    # 7. 找到边缘位置（R40交叉点）
    binary_signal = np.array(reflectance_profiles) < R40
    edges = np.where(np.diff(binary_signal.astype(int)))[0]

    # 8. 错误处理：必须找到成对的左右边缘
    if len(edges) < 2:
        raise ValueError("无法检测到有效的边缘对")

    # 9. 转换为物理尺寸（1200 dpi = 47.244像素/毫米）
    pixels_per_mm = 47.244
    left_edge_px = edges[0] + scan_start
    right_edge_px = edges[-1] + scan_start
    line_width_mm = (right_edge_px - left_edge_px) / pixels_per_mm

    # 10. 在图像上标记边缘（垂直线标记左右边缘）
    marked_img = image.copy()
    cv2.line(marked_img,
             (left_edge_px, 0), (left_edge_px, height),
             (0, 0, 255), 2)  # 左边缘（红色）
    cv2.line(marked_img,
             (right_edge_px, 0), (right_edge_px, height),
             (0, 255, 0), 2)  # 右边缘（绿色）

    return (
        round(line_width_mm, 4),  # 线宽(mm)
        marked_img,  # 标记图像
        reflectance_profiles,  # 反射率曲线
        R40,  # 阈值
        scan_start,  # 扫描起始
        scan_end  # 扫描结束
    )


def measure_roi_attributes(image_path, dpi=1200, debug=False):
    """
    测量ROI图像中的反射率阈值和线宽（符合ISO/IEC 24790-2017附录D）
    :param image_path: 输入图像路径
    :param dpi: 图像分辨率（每英寸点数，默认1200 DPI）
    :param debug: 是否显示中间处理结果
    :return: 阈值字典、线宽（毫米）、字符暗度
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

    # 4. 估计线宽（使用初步方法）
    # 这里可以使用一个简单的初步方法来估计线宽
    # 例如，使用扫描线方法初步估计线宽
    binary_image = cv2.threshold(rotated_gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
    estimated_line_width_px = measure_line_width(binary_image, num_samples=5)
    estimated_line_width_mm = (estimated_line_width_px / dpi) * 25.4

    # 5. 设置ROI的宽度和高度（单位：毫米）
    roi_width_mm = max(estimated_line_width_mm + 2, 2)  # 至少比估计线宽宽2 mm
    roi_height_mm = max(5, 5)  # 不少于5 mm

    # 将ROI尺寸从毫米转换为像素
    roi_width_px = int((roi_width_mm / 25.4) * dpi)
    roi_height_px = int((roi_height_mm / 25.4) * dpi)

    # 以线条中心为中心裁剪ROI
    x1 = max(cx - roi_width_px // 2, 0)
    y1 = max(cy - roi_height_px // 2, 0)
    x2 = min(x1 + roi_width_px, rotated_gray.shape[1])
    y2 = min(y1 + roi_height_px, rotated_gray.shape[0])
    roi = rotated_gray[y1:y2, x1:x2]

    # 6. 自动检测背景和线条区域（双峰法）
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    peaks, _ = find_peaks(hist.flatten(), height=50, distance=50)
    if not validate_peaks(peaks, hist):
        raise ValueError("反射率分布不呈现有效的双峰形状")

    # 7. 将灰度值归一化到[0, 1]范围（反射率）
    r_min = peaks[0] / 255.0
    r_max = peaks[1] / 255.0

    # 8. 计算阈值（R10, R40, R70, R75）
    r10 = r_min + 0.1 * (r_max - r_min)
    r40 = r_min + 0.4 * (r_max - r_min)
    r70 = r_min + 0.7 * (r_max - r_min)
    r75 = r_min + 0.75 * (r_max - r_min)


    # 9. 二值化并提取线条轮廓
    _, binary = cv2.threshold(roi, int(r40 * 255), 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到线条")

    # 取最大轮廓（假设为待测线条）
    main_contour = max(contours, key=cv2.contourArea)

    # 10. 计算线条的最小外接矩形
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

    # 11. 使用公式（19）计算线宽
    line_width_px = measure_line_width(binary, num_samples=10)

    # 12. 将线宽从像素转换为毫米
    line_width_mm = (line_width_px / dpi) * 25.4

    # 13. 计算线条图像密度
    r25 = r_min + 0.25 * (r_max - r_min)
    line_density = -np.log10(r25)

    # 14. 计算字符暗度
    character_darkness = line_density * np.sqrt(line_width_mm)

    # 15. 使用狭缝扫描法验证线宽（使用旋转后的图像）
    # 将旋转后的灰度图转换为BGR格式供标记使用
    rotated_bgr = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
    slit_width_mm, marked_img, reflectance_profiles, R40_slit, scan_start, scan_end = calculate_line_width_slit(
        rotated_bgr)

    # 16. 可视化结果
    if debug:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
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
        ax[0, 2].imshow(overlay)
        ax[0, 2].set_title("边缘检测与线宽测量")
        ax[0, 2].axis('off')

        ax[1, 0].axis('off')
        text = f"""
        R_max (背景): {r_max:.2f}
        R_min (线条): {r_min:.2f}
        R10 (内边界): {r10:.2f}
        R40 (边缘阈值): {r40:.2f}
        R70 (外边界): {r70:.2f}
        R75: {r75:.2f}
        线宽: {line_width_mm:.5f} mm
        线条图像密度: {line_density:.2f}
        字符暗度: {character_darkness:.2f}
        狭缝法线宽: {slit_width_mm:.5f} mm
        """
        ax[1, 0].text(0.1, 0.5, text, fontsize=12)

        # 显示狭缝扫描结果
        ax[1, 1].plot(range(scan_start, scan_end), reflectance_profiles, label='反射率曲线')
        ax[1, 1].axhline(R40_slit, color='r', linestyle='--', label=f'R40阈值 ({R40_slit:.3f})')
        ax[1, 1].set_xlabel('水平位置（像素）')
        ax[1, 1].set_ylabel('反射率')
        ax[1, 1].set_title('狭缝扫描反射率曲线')
        ax[1, 1].legend()

        # 显示标记图像
        ax[1, 2].imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        ax[1, 2].set_title('狭缝法边缘检测')
        ax[1, 2].axis('off')

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
        "SlitLineWidth_mm": slit_width_mm,
        "LineDensity": line_density,
        "CharacterDarkness": character_darkness
    }



# 示例调用
if __name__ == "__main__":
    # 测试图像路径
    test_image_path = "D:/PycharmProjects/pythonProject/ISO24790/output_rois23/L13_Mark.tif"
    dpi = 1200  # 图像分辨率（DPI）

    # 测量ROI属性
    results = measure_roi_attributes(test_image_path, dpi=dpi, debug=True)

    # 打印结果
    print("\n测量结果:")
    print(f"背景最大反射率 (R_max): {results['R_max']:.4f}")
    print(f"线条最小反射率 (R_min): {results['R_min']:.4f}")
    print(f"内边界阈值 (R10): {results['R10']:.4f}")
    print(f"边缘阈值 (R40): {results['R40']:.4f}")
    print(f"外边界阈值 (R70): {results['R70']:.4f}")
    print(f"线宽 (常规方法): {results['LineWidth_mm']:.5f} mm")
    print(f"线宽 (狭缝法): {results['SlitLineWidth_mm']:.5f} mm")
    print(f"线条图像密度: {results['LineDensity']:.4f}")
    print(f"字符暗度: {results['CharacterDarkness']:.4f}")