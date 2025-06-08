import cv2
import numpy as np
import csv
from pathlib import Path

class ROIExtractor:
    def __init__(self, config_path, dpi=1200):
        self.dpi = dpi
        self.mm_to_pixel = dpi / 25.4  # 毫米转像素系数
        self.fiducials = {}  # 存储基准点像素坐标
        self.measurements = []  # 存储测量区域定义
        self.calibrations = {}  # 存储校准参数
        self._parse_config(config_path)  # 解析配置文件
        self.window_name = "Fiducial Verification"
        self.rotation = 0  # 初始旋转角度
        self.cursor_pos = (200, 173)  # 初始光标位置
        self.mark_pos = None  # 注册标记位置
        self.dragging = False  # 是否正在拖动光标
        self.rotation_map = {
            0: {'ULC': 'ULC', 'URC': 'URC', 'LRC': 'LRC', 'LLC': 'LLC'},
            90: {'ULC': 'LLC', 'URC': 'ULC', 'LRC': 'URC', 'LLC': 'LRC'},
            180: {'ULC': 'LRC', 'URC': 'LLC', 'LRC': 'ULC', 'LLC': 'URC'},
            270: {'ULC': 'URC', 'URC': 'LRC', 'LRC': 'LLC', 'LLC': 'ULC'}
        }
        # 初始化窗口和回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event)

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
                            print(f"Parsing {row[1].strip()}: mm=({mm_x}, {mm_y}) -> pixels=({pixel_x}, {pixel_y})")
                            self.fiducials[row[1].strip()] = (pixel_x, pixel_y)
                    elif current_section == 'MEASUREMENT' and len(row) >= 6:
                        if all(row[1:6]):
                            # 将毫米坐标转换为像素坐标
                            mm_x = float(row[2].strip())
                            mm_y = float(row[3].strip())
                            mm_dx = float(row[4].strip())
                            mm_dy = float(row[5].strip())
                            pixel_x = int(mm_x * self.mm_to_pixel)
                            pixel_y = int(mm_y * self.mm_to_pixel)
                            pixel_dx = int(mm_dx * self.mm_to_pixel)
                            pixel_dy = int(mm_dy * self.mm_to_pixel)
                            self.measurements.append({
                                'id': row[1].strip(),
                                'x': pixel_x, 'y': pixel_y,
                                'dx': pixel_dx, 'dy': pixel_dy
                            })
                    elif current_section == 'CALIBRATION' and len(row) >= 9:
                        if all(row[1:9]):
                            # 解析校准数据
                            self.calibrations[row[1].strip()] = {
                                'x': float(row[2].strip()),  # X 坐标
                                'y': float(row[3].strip()),  # Y 坐标
                                'dx': float(row[4].strip()),  # dX 宽度
                                'dy': float(row[5].strip()),  # dY 高度
                                'dvis': float(row[6].strip()),  # Dvis 值
                                'cie_x': float(row[7].strip()),  # CIE_X 值
                                'cie_y': float(row[8].strip()),  # CIE_Y 值
                                'cie_z': float(row[9].strip())  # CIE_Z 值
                            }
                except (ValueError, IndexError) as e:
                    print(f"解析错误：{row}，错误：{e}")

    def _draw_axes(self, image):
        """绘制坐标轴和刻度"""
        h, w = image.shape[:2]
        cv2.line(image, (0, h - 1), (w, h - 1), (0, 0, 255), 2)  # X轴
        cv2.line(image, (0, 0), (0, h), (0, 0, 255), 2)  # Y轴
        for x in range(50, w, 50):
            cv2.line(image, (x, h - 1), (x, h - 10), (0, 0, 255), 1)
            cv2.putText(image, f"{x}", (x - 20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for y in range(50, h, 50):
            cv2.line(image, (0, y), (10, y), (0, 0, 255), 1)
            cv2.putText(image, f"{y}", (15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _draw_cursor(self, image):
        """绘制光标位置"""
        x, y = self.cursor_pos
        cv2.line(image, (x - 20, y), (x + 20, y), (0, 255, 0), 2)  # 水平线
        cv2.line(image, (x, y - 20), (x, y + 20), (0, 255, 0), 2)  # 垂直线
        cv2.putText(image, f"Cursor: ({x}, {y})", (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_mark(self, image):
        """绘制注册标记"""
        if self.mark_pos is not None:
            x, y = self.mark_pos
            cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=2)
            cv2.putText(image, f"Mark: ({x}, {y})", (x + 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _draw_control_panel(self, image):
        """绘制右侧控制面板（包含滑块）"""
        h, w = image.shape[:2]
        panel_w = 200
        cv2.rectangle(image, (w - panel_w, 0), (w, h), (200, 200, 200), -1)

        # 显示光标坐标
        y_offset = 30
        cv2.putText(image, "Cursor Setting", (w - panel_w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 40
        cv2.putText(image, f"X: {self.cursor_pos[0]}", (w - panel_w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)
        y_offset += 30
        cv2.putText(image, f"Y: {self.cursor_pos[1]}", (w - panel_w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)
        y_offset += 50

        # 绘制绿色OK按钮
        btn_color = (0, 180, 0)
        cv2.rectangle(image, (w - panel_w + 20, y_offset), (w - 20, y_offset + 40), btn_color, -1)
        cv2.putText(image, "OK", (w - panel_w + 80, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _mouse_event(self, event, x, y, flags, param):
        """鼠标事件处理（同步滑块位置）"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.cursor_pos = (x, y)
            cv2.setTrackbarPos("X", self.window_name, x)
            cv2.setTrackbarPos("Y", self.window_name, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cursor_pos = (x, y)
            cv2.setTrackbarPos("X", self.window_name, x)
            cv2.setTrackbarPos("Y", self.window_name, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mark_pos = (x, y)

    def _visualize_fiducials(self, image):
        """实时刷新的可视化界面"""
        display = image.copy()

        # 初始化滑块（仅一次）
        cv2.createTrackbar("X", self.window_name, self.cursor_pos[0], display.shape[1], lambda x: None)
        cv2.createTrackbar("Y", self.window_name, self.cursor_pos[1], display.shape[0], lambda y: None)

        while True:
            # 更新光标位置（从滑块读取）
            x = cv2.getTrackbarPos("X", self.window_name)
            y = cv2.getTrackbarPos("Y", self.window_name)
            self.cursor_pos = (x, y)

            # 绘制界面
            display_copy = display.copy()
            self._draw_axes(display_copy)
            self._draw_cursor(display_copy)
            self._draw_mark(display_copy)
            self._draw_control_panel(display_copy)

            # 显示并等待按键
            cv2.imshow(self.window_name, display_copy)
            key = cv2.waitKey(10)

            if key == 27:  # ESC键退出
                cv2.destroyAllWindows()
                return False
            elif key == 32:  # 空格键确认
                cv2.destroyAllWindows()
                return True
            elif cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:  # 窗口被关闭
                return False

    def process_image(self, image_path, output_dir):
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")

        # 步骤1: 可视化并设置 cursor 为 ULC
        if not self._visualize_fiducials(img):
            print("处理已取消")
            return

        # 步骤2: 透视变换
        # 获取基准点坐标
        src_points = np.array([
            self.fiducials['ULC'], self.fiducials['URC'],
            self.fiducials['LRC'], self.fiducials['LLC']
        ], dtype=np.float32)

        # 计算目标图像的宽度和高度
        width = int(np.linalg.norm(src_points[1] - src_points[0]))  # URC 到 ULC 的距离
        height = int(np.linalg.norm(src_points[2] - src_points[1]))  # LRC 到 URC 的距离

        # 定义目标点
        dst_points = np.array([
            [0, 0], [width, 0],
            [width, height], [0, height]
        ], dtype=np.float32)

        # 计算透视变换矩阵并应用变换
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)

        # 步骤3: 提取 ROI
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for meas in self.measurements:
            # 获取 Measurement 的坐标和尺寸
            x = meas['x']
            y = meas['y']
            dx = meas['dx']
            dy = meas['dy']

            # 计算 ROI 区域
            x1 = x - dx // 2  # 左上角 x
            y1 = y - dy // 2  # 左上角 y
            x2 = x + dx // 2  # 右下角 x
            y2 = y + dy // 2  # 右下角 y

            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(warped.shape[1], x2)
            y2 = min(warped.shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                print(f"警告：{meas['id']} ROI 超出图像边界，已跳过")
                continue

            # 提取 ROI
            roi = warped[y1:y2, x1:x2]

            # 步骤4: 使用 Calibration 调整 ROI 亮度
            calibrated_roi = self.apply_calibration(roi, meas['id'])

            # 步骤5: 保存 ROI
            cv2.imwrite(str(output_dir / f"{meas['id']}.tif"), calibrated_roi)

    def apply_calibration(self, roi, meas_id):
        """应用校准参数（使用视觉反射率 Rv 调整亮度）"""
        if meas_id in self.calibrations:
            calib = self.calibrations[meas_id]

            # 获取 Calibration 的坐标和尺寸
            calib_x = int(calib['x'] * self.mm_to_pixel)
            calib_y = int(calib['y'] * self.mm_to_pixel)
            calib_dx = int(calib['dx'] * self.mm_to_pixel)
            calib_dy = int(calib['dy'] * self.mm_to_pixel)

            # 计算 Calibration 的 ROI 区域
            calib_x1 = calib_x - calib_dx // 2
            calib_y1 = calib_y - calib_dy // 2
            calib_x2 = calib_x + calib_dx // 2
            calib_y2 = calib_y + calib_dy // 2

            # 边界检查
            calib_x1 = max(0, calib_x1)
            calib_y1 = max(0, calib_y1)
            calib_x2 = min(roi.shape[1], calib_x2)
            calib_y2 = min(roi.shape[0], calib_y2)

            if calib_x1 >= calib_x2 or calib_y1 >= calib_y2:
                print(f"警告：{meas_id} 的 Calibration ROI 超出边界，跳过亮度调整")
                return roi

            # 提取 Calibration 的 ROI
            calib_roi = roi[calib_y1:calib_y2, calib_x1:calib_x2]

            # 计算视觉反射率 Rv
            dvis = calib['dvis']  # 获取 Dvis 值
            rv = 10 ** (-dvis)  # 计算 Rv = 10^(-Dv)

            # 将 Rv 作为亮度调整的缩放因子
            scale = rv

            # 调整 ROI 的亮度
            calib_roi = calib_roi.astype(np.float32)  # 转换为 float32
            calib_roi = np.clip(calib_roi * scale, 0, 255)  # 调整亮度并截断
            calib_roi = calib_roi.astype(np.uint8)  # 转换回 uint8

            # 将调整后的 Calibration ROI 复制回原始 ROI
            roi[calib_y1:calib_y2, calib_x1:calib_x2] = calib_roi

        return roi


if __name__ == "__main__":
    # 使用示例
    extractor = ROIExtractor(
        config_path="C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/data/Config_Conformance2021Pro_Half_Calib_XYZ.txt",
        dpi=1200
    )
    extractor.process_image(
        "C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/Sample_img/ES10000005_1200_Image2_1.tif",
        "output_rois23"
    )