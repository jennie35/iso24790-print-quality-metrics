import cv2
import numpy as np
import csv
from pathlib import Path

class ROIExtractor:
    def __init__(self, config_path, dpi=1200):
        self.dpi = dpi
        self.mm_to_pixel = dpi / 25.4  # 毫米到像素转换系数
        self.fiducials = {}             # 存储基准点像素坐标
        self.measurements = []          # 存储测量区域定义
        self.calibrations = {}          # 存储校准参数
        self._parse_config(config_path) # 解析配置文件
        self.window_name = "Fiducial Verification"
        self.rotation = 0               # 初始旋转角度
        # 定义不同旋转角度下的基准点位置映射
        self.rotation_map = {
            0: {'ULC': 'ULC', 'URC': 'URC', 'LRC': 'LRC', 'LLC': 'LLC'},
            90: {'ULC': 'LLC', 'URC': 'ULC', 'LRC': 'URC', 'LLC': 'LRC'},
            180: {'ULC': 'LRC', 'URC': 'LLC', 'LRC': 'ULC', 'LLC': 'URC'},
            270: {'ULC': 'URC', 'URC': 'LRC', 'LRC': 'LLC', 'LLC': 'ULC'}
        }

    def _parse_config(self, config_path):
        """解析配置文件，处理Fiducials、Measurement、Calibration三个部分"""
        current_section = None
        with open(config_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or all(cell.strip() == '' for cell in row):
                    continue  # 跳过空行

                # 检测章节标题
                if row[0] == 'Fiducials':
                    current_section = 'FIDUCIALS'
                    continue
                elif row[0] == 'Measurement':
                    current_section = 'MEASUREMENT'
                    continue
                elif row[0] == 'Calibration':
                    current_section = 'CALIBRATION'
                    continue

                try:
                    if current_section == 'FIDUCIALS' and len(row) >= 4:
                        if row[1].strip() and row[2].strip() and row[3].strip():
                            # 将毫米坐标转换为像素坐标
                            mm_x = float(row[2].strip())
                            mm_y = float(row[3].strip())
                            pixel_x = int(mm_x * self.mm_to_pixel)
                            pixel_y = int(mm_y * self.mm_to_pixel)
                            self.fiducials[row[1].strip()] = (pixel_x, pixel_y)
                    elif current_section == 'MEASUREMENT' and len(row) >= 6:
                        if all(row[1:6]):
                            self.measurements.append({
                                'id': row[1].strip(),
                                'x': float(row[2].strip()),   # 单位：毫米
                                'y': float(row[3].strip()),
                                'dx': float(row[4].strip()),  # ROI宽度
                                'dy': float(row[5].strip())   # ROI高度
                            })
                    elif current_section == 'CALIBRATION' and len(row) >= 9:
                        if all(row[1:9]):
                            self.calibrations[row[1].strip()] = {
                                'dvis': float(row[5].strip()),
                                'cie_x': float(row[6].strip()),
                                'cie_y': float(row[7].strip()),
                                'cie_z': float(row[8].strip())
                            }
                except (ValueError, IndexError) as e:
                    print(f"解析错误：{row}，错误：{e}")

    def _visualize_fiducials(self, image, points):
        """可视化基准点并等待用户确认"""
        display = image.copy()
        colors = {
            'ULC': (0, 255, 0),    # 绿色
            'URC': (0, 200, 255),  # 橙色
            'LRC': (255, 0, 0),    # 蓝色
            'LLC': (255, 0, 255)   # 紫色
        }

        # 绘制基准点和坐标信息
        for fid_id, (x, y) in points.items():
            color = colors.get(fid_id, (0, 0, 0))
            cv2.drawMarker(display, (x, y), color,
                         markerType=cv2.MARKER_CROSS,
                         markerSize=40, thickness=2)
            cv2.putText(display, f"{fid_id} ({x},{y})", (x + 45, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示操作提示
        cv2.putText(display, f"Rotation: {self.rotation}°", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Space: Confirm | ESC: Cancel", (20, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key == 32  # 空格键确认

    def _detect_rotation(self, detected_points):
        """根据基准点位置自动检测旋转角度"""
        # 计算基准点中心坐标
        points = np.array(list(detected_points.values()))
        center = np.mean(points, axis=0)

        # 确定每个基准点的象限位置
        position_map = {}
        for fid_id, (x, y) in detected_points.items():
            if x < center[0] and y < center[1]:
                position_map['ULC'] = fid_id  # 左上象限
            elif x > center[0] and y < center[1]:
                position_map['URC'] = fid_id  # 右上象限
            elif x > center[0] and y > center[1]:
                position_map['LRC'] = fid_id  # 右下象限
            else:
                position_map['LLC'] = fid_id  # 左下象限

        # 匹配预定义的旋转映射
        for angle, mapping in self.rotation_map.items():
            if all(position_map[k] == v for k, v in mapping.items()):
                self.rotation = angle
                return
        raise ValueError("无法确定旋转角度，请检查基准点位置")

    def _rotate_image(self, image):
        """执行图像旋转校正"""
        if self.rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _get_transformed_points(self):
        """获取旋转后的基准点坐标"""
        rotated_points = {}
        for std_id, orig_id in self.rotation_map[self.rotation].items():
            rotated_points[std_id] = self.fiducials[orig_id]
        return rotated_points

    def process_image(self, image_path, output_dir):
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")

        # 步骤1: 检测旋转角度
        self._detect_rotation(self.fiducials)

        # 步骤2: 执行图像旋转
        rotated_img = self._rotate_image(img)

        # 步骤3: 获取旋转后的基准点坐标
        adjusted_points = self._get_transformed_points()

        # 步骤4: 可视化验证基准点
        if not self._visualize_fiducials(rotated_img, adjusted_points):
            print("处理已取消")
            return

        # 步骤5: 透视变换
        src_points = np.array([
            adjusted_points['ULC'], adjusted_points['URC'],
            adjusted_points['LRC'], adjusted_points['LLC']
        ], dtype=np.float32)

        # 计算目标尺寸（基于基准点物理位置）
        ulc = np.array(adjusted_points['ULC'])
        urc = np.array(adjusted_points['URC'])
        llc = np.array(adjusted_points['LLC'])
        target_width = int(np.linalg.norm(urc - ulc))  # 计算URC到ULC的欧氏距离
        target_height = int(np.linalg.norm(llc - ulc)) # 计算LLC到ULC的欧氏距离

        dst_points = np.array([
            [0, 0], [target_width, 0],
            [target_width, target_height], [0, target_height]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(rotated_img, matrix,
                                   (target_width, target_height),
                                   flags=cv2.INTER_CUBIC)

        # 步骤6: 提取ROI
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for meas in self.measurements:
            # 将毫米坐标转换为像素坐标
            x = int(meas['x'] * self.mm_to_pixel)
            y = int(meas['y'] * self.mm_to_pixel)
            dx = int(meas['dx'] * self.mm_to_pixel)
            dy = int(meas['dy'] * self.mm_to_pixel)

            # 计算ROI边界（中心坐标转换为矩形区域）
            x1 = max(0, x - dx // 2)
            y1 = max(0, y - dy // 2)
            x2 = min(warped.shape[1], x + dx // 2)
            y2 = min(warped.shape[0], y + dy // 2)

            if x1 >= x2 or y1 >= y2:
                print(f"警告：{meas['id']} ROI超出图像边界，已跳过")
                continue

            # 提取并保存ROI
            roi = warped[y1:y2, x1:x2]
            calibrated_roi = self.apply_calibration(roi, meas['id'])
            cv2.imwrite(str(output_dir / f"{meas['id']}.tif"), calibrated_roi)

    def apply_calibration(self, roi, meas_id):
        """应用校准参数（示例：亮度调整）"""
        if meas_id in self.calibrations:
            calib = self.calibrations[meas_id]
            # 示例：根据CIE Y值调整亮度
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness > 0:
                scale = calib['cie_y'] / avg_brightness
                roi = np.clip(roi.astype(np.float32) * scale, 0, 255)
            return roi.astype(np.uint8)
        return roi

if __name__ == "__main__":
    # 使用示例
    extractor = ROIExtractor(
        config_path="C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/data/Config_Conformance2021Pro_Half_Calib_XYZ.txt",
        dpi=1200
    )
    extractor.process_image(
        image_path="C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000005_1200_Image2_1.tif",
        output_dir="output_rois21"
    )