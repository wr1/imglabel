"""Graphical user interface for imglabel using PyQt6."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage, QPolygonF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QRadioButton, QLineEdit, QSlider, QLabel, QButtonGroup, QFileDialog, QMessageBox,
    QGraphicsItem, QGridLayout
)
from skimage.color import rgb2gray
from skimage.feature import canny

from .core import (
    load_and_coarsen_image,
    compute_similar_mask,
    compute_mixed_mask,
    cluster_and_get_polygons,
    detect_lines,
    get_line_boundary_points,
    save_criteria,
    load_criteria,
    select_boundary_component,
)


class ImageGraphicsView(QGraphicsView):
    """Custom graphics view for image display with zooming and clicking."""
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.zoom_factor = 1.0
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.polygons = []
        self.lines = []
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_image(self, pixmap):
        """Set the image pixmap."""
        self.scene.clear()
        # Clear the lists to avoid holding references to deleted items
        self.polygons.clear()
        self.lines.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def add_polygon(self, points, color):
        """Add a polygon overlay."""
        poly = QGraphicsPolygonItem()
        qpoints = [QPointF(p[0], p[1]) for p in points]
        poly.setPolygon(QPolygonF(qpoints))
        poly.setPen(QPen(color, 2))
        poly.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.scene.addItem(poly)
        self.polygons.append(poly)
        poly.setZValue(1)

    def clear_polygons(self):
        """Clear all polygons."""
        for poly in self.polygons:
            self.scene.removeItem(poly)
        self.polygons.clear()

    def add_line(self, start, end, color):
        """Add a line overlay."""
        line = self.scene.addLine(start[0], start[1], end[0], end[1], QPen(color, 2))
        self.lines.append(line)
        line.setZValue(1)
        return line

    def clear_lines(self):
        """Clear all lines."""
        for line in self.lines:
            self.scene.removeItem(line)
        self.lines.clear()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        """Zoom in."""
        self.zoom_factor *= 1.1
        self.zoom_factor = min(5.0, self.zoom_factor)
        self.update_zoom()

    def zoom_out(self):
        """Zoom out."""
        self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, self.zoom_factor)
        self.update_zoom()

    def update_zoom(self):
        """Update the zoom level."""
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self.pixmap_item:
            pos = self.mapToScene(event.pos())
            rect = self.pixmap_item.boundingRect()
            if rect.contains(pos):
                x = pos.x()
                y = pos.y()
                self.clicked.emit(x, y)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)


class ImageLabelerGUI(QMainWindow):
    """Main GUI class for image labeling."""
    def __init__(self, dir_path=None):
        super().__init__()
        self.selected_hsv = None
        self.selected_shape = None
        self.threshold_hue = 0.05
        self.threshold_sat = 0.2
        self.threshold_shape = 0.5
        self.img_coarsened = None
        self.img_hsv = None
        self.dir_path = dir_path
        self.mask_points = None
        self.clicked_x = None
        self.clicked_y = None
        self.lines = []
        self.selected_lines = []
        self.line_sensitivity = 1.5
        self.current_image_path = None
        self.setup_ui()
        if self.dir_path:
            self.load_directory(self.dir_path)
            self.select_first_image()
        self.showFullScreen()

    def setup_ui(self):
        """Set up the user interface components."""
        self.setWindowTitle("imglabel GUI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.save_coco_btn = QPushButton("Save COCO")
        self.save_coco_btn.clicked.connect(self.save_coco)
        left_layout.addWidget(self.save_coco_btn)
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_select)
        left_layout.addWidget(self.file_list)
        main_layout.addWidget(left_panel, 1)

        # Middle: Graphics view
        self.graphics_view = ImageGraphicsView()
        self.graphics_view.clicked.connect(self.on_canvas_click)
        main_layout.addWidget(self.graphics_view, 3)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Filter:"))
        filter_layout = QHBoxLayout()
        self.filter_group = QButtonGroup(self)
        self.color_radio = QRadioButton("Color")
        self.color_radio.setChecked(True)
        self.filter_group.addButton(self.color_radio)
        filter_layout.addWidget(self.color_radio)
        self.mixed_radio = QRadioButton("Mixed")
        self.filter_group.addButton(self.mixed_radio)
        filter_layout.addWidget(self.mixed_radio)
        self.line_radio = QRadioButton("Line")
        self.filter_group.addButton(self.line_radio)
        filter_layout.addWidget(self.line_radio)
        right_layout.addLayout(filter_layout)
        self.filter_group.buttonClicked.connect(self.on_filter_change)

        settings_frame = QWidget()
        settings_layout = QGridLayout(settings_frame)
        settings_layout.addWidget(QLabel("Settings:"), 0, 0, 1, 2)
        self.hue_label = QLabel("Hue:")
        settings_layout.addWidget(self.hue_label, 1, 0)
        self.hue_entry = QLineEdit(str(self.threshold_hue))
        self.hue_entry.returnPressed.connect(self.update_hue_threshold)
        settings_layout.addWidget(self.hue_entry, 1, 1)
        self.sat_label = QLabel("Sat:")
        settings_layout.addWidget(self.sat_label, 2, 0)
        self.sat_entry = QLineEdit(str(self.threshold_sat))
        self.sat_entry.returnPressed.connect(self.update_sat_threshold)
        settings_layout.addWidget(self.sat_entry, 2, 1)
        self.shape_label = QLabel("Shape:")
        settings_layout.addWidget(self.shape_label, 3, 0)
        self.shape_entry = QLineEdit(str(self.threshold_shape))
        self.shape_entry.returnPressed.connect(self.update_shape_threshold)
        settings_layout.addWidget(self.shape_entry, 3, 1)
        self.sensitivity_label = QLabel("Sensitivity:")
        settings_layout.addWidget(self.sensitivity_label, 4, 0)
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 30)
        self.sensitivity_slider.setValue(int(self.line_sensitivity * 10))
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        settings_layout.addWidget(self.sensitivity_slider, 4, 1)
        self.mask_btn = QPushButton("MASK")
        self.mask_btn.clicked.connect(self.set_mask)
        settings_layout.addWidget(self.mask_btn, 5, 0, 1, 2)
        right_layout.addWidget(settings_frame)
        main_layout.addWidget(right_panel, 1)

        self.on_filter_change()

    def save_coco(self):
        """Save current polygons to COCO format."""
        if not self.current_image_path or self.img_coarsened is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save COCO", "", "JSON files (*.json)")
        if not file_path:
            return
        height, width = self.img_coarsened.shape[:2]
        coco = {
            "info": {},
            "licenses": [],
            "images": [{
                "id": 1,
                "file_name": self.current_image_path,
                "height": height,
                "width": width
            }],
            "annotations": [],
            "categories": [{"id": 1, "name": "damage"}]
        }
        ann_id = 1
        for poly in self.graphics_view.polygons:
            qpoly = poly.polygon()
            points = []
            for i in range(qpoly.size()):
                p = qpoly.at(i)
                points.extend([p.x(), p.y()])
            if len(points) < 6:  # at least 3 points
                continue
            # compute bbox
            xs = points[::2]
            ys = points[1::2]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            # compute area using shoelace
            area = 0
            n = len(xs)
            for i in range(n):
                j = (i + 1) % n
                area += xs[i] * ys[j] - xs[j] * ys[i]
            area = abs(area) / 2
            annotation = {
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [points],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco["annotations"].append(annotation)
            ann_id += 1
        with open(file_path, 'w') as f:
            json.dump(coco, f, indent=2)
        QMessageBox.information(self, "Saved", f"COCO saved to {file_path}")

    def on_filter_change(self):
        """Handle filter change."""
        self.selected_lines = []
        self.selected_hsv = None
        self.selected_shape = None
        filter_type = self.filter_group.checkedButton().text()
        # Hide/show settings
        visible = filter_type in ("Color", "Mixed")
        self.hue_label.setVisible(visible)
        self.hue_entry.setVisible(visible)
        self.sat_label.setVisible(visible)
        self.sat_entry.setVisible(visible)
        self.shape_label.setVisible(filter_type == "Mixed")
        self.shape_entry.setVisible(filter_type == "Mixed")
        self.sensitivity_label.setVisible(filter_type == "Line")
        self.sensitivity_slider.setVisible(filter_type == "Line")
        self.mask_btn.setVisible(filter_type == "Line")
        self.display_image()

    def update_hue_threshold(self):
        """Update hue threshold."""
        try:
            self.threshold_hue = float(self.hue_entry.text())
            self.update_display()
        except ValueError:
            self.hue_entry.setText(str(self.threshold_hue))

    def update_sat_threshold(self):
        """Update saturation threshold."""
        try:
            self.threshold_sat = float(self.sat_entry.text())
            self.update_display()
        except ValueError:
            self.sat_entry.setText(str(self.threshold_sat))

    def update_shape_threshold(self):
        """Update shape threshold."""
        try:
            self.threshold_shape = float(self.shape_entry.text())
            self.update_display()
        except ValueError:
            self.shape_entry.setText(str(self.threshold_shape))

    def update_sensitivity(self, value):
        """Update sensitivity."""
        self.line_sensitivity = value / 10.0
        self.display_image()

    def set_mask(self):
        """Set the mask from selected lines."""
        if len(self.selected_lines) == 2:
            idx1, idx2 = self.selected_lines
            rho1, theta1 = self.lines[idx1]
            rho2, theta2 = self.lines[idx2]
            height, width = self.img_coarsened.shape[:2]
            points1 = get_line_boundary_points(rho1, theta1, width, height)
            points2 = get_line_boundary_points(rho2, theta2, width, height)
            if len(points1) == 2 and len(points2) == 2:
                self.mask_points = points1 + list(reversed(points2))
                QMessageBox.information(self, "Mask Set", "Mask set from selected lines.")
        else:
            QMessageBox.warning(self, "No Selection", "Select two lines first.")

    def load_directory(self, dir_path):
        """Load image files."""
        self.dir_path = dir_path
        self.file_list.clear()
        for f in Path(dir_path).iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                self.file_list.addItem(f.name)

    def select_first_image(self):
        """Select first image."""
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
            self.on_file_select()

    def on_file_select(self):
        """Handle file selection."""
        item = self.file_list.currentItem()
        if item:
            filename = item.text()
            self.load_image(os.path.join(self.dir_path, filename))

    def load_image(self, path):
        """Load and display image."""
        try:
            self.img_coarsened, self.img_hsv = load_and_coarsen_image(path)
            self.current_image_path = path
            self.selected_hsv = None
            self.selected_shape = None
            self.selected_lines = []
            self.mask_points = None
            self.clicked_x = None
            self.clicked_y = None
            self.display_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def display_image(self):
        """Display the image."""
        if self.img_coarsened is None:
            return
        img = Image.fromarray(self.img_coarsened)
        width, height = img.size
        if self.mask_points and self.filter_group.checkedButton().text() in ("Color", "Mixed"):
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(self.mask_points, fill=255)
            gray_img = Image.new('RGB', (width, height), (211, 211, 211))
            img_display = Image.composite(img, gray_img, mask_img)
        else:
            img_display = img
        pixmap = QPixmap.fromImage(QImage(img_display.tobytes(), width, height, QImage.Format.Format_RGB888))
        self.graphics_view.set_image(pixmap)
        self.graphics_view.zoom_factor = 1.0
        self.graphics_view.update_zoom()
        filter_type = self.filter_group.checkedButton().text()
        if filter_type == "Line":
            img_gray = rgb2gray(self.img_coarsened)
            self.lines = detect_lines(img_gray, sigma=self.line_sensitivity)
            self.draw_lines()
        self.update_display()

    def draw_lines(self):
        """Draw detected lines."""
        self.graphics_view.clear_lines()
        for i, (rho, theta) in enumerate(self.lines):
            points = get_line_boundary_points(rho, theta, self.img_coarsened.shape[1], self.img_coarsened.shape[0])
            if len(points) == 2:
                start = (points[0][0], points[0][1])
                end = (points[1][0], points[1][1])
                color = QColor('red') if i in self.selected_lines else QColor('blue')
                self.graphics_view.add_line(start, end, color)

    def on_canvas_click(self, x, y):
        """Handle canvas click."""
        if self.img_hsv is None:
            return
        height, width = self.img_coarsened.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        filter_type = self.filter_group.checkedButton().text()
        if filter_type in ("Color", "Mixed"):
            half_size = 2
            x_start = max(0, int(x) - half_size)
            x_end = min(width, int(x) + half_size + 1)
            y_start = max(0, int(y) - half_size)
            y_end = min(height, int(y) + half_size + 1)
            patch_hsv = self.img_hsv[y_start:y_end, x_start:x_end]
            patch_rgb = self.img_coarsened[y_start:y_end, x_start:x_end]
            if patch_hsv.size == 0:
                return
            h_patch = patch_hsv[:, :, 0]
            s_patch = patch_hsv[:, :, 1]
            v_patch = patch_hsv[:, :, 2]
            sin_h = np.sin(2 * np.pi * h_patch)
            cos_h = np.cos(2 * np.pi * h_patch)
            mean_sin = np.mean(sin_h)
            mean_cos = np.mean(cos_h)
            mean_h = np.arctan2(mean_sin, mean_cos) / (2 * np.pi)
            if mean_h < 0:
                mean_h += 1
            mean_s = np.mean(s_patch)
            mean_v = np.mean(v_patch)
            self.selected_hsv = np.array([mean_h, mean_s, mean_v])
            self.clicked_x = int(x)
            self.clicked_y = int(y)
            if filter_type == "Mixed":
                patch_gray = rgb2gray(patch_rgb)
                edges = canny(patch_gray, sigma=1.0)
                self.selected_shape = np.mean(edges)
            self.update_display()
        elif filter_type == "Line":
            min_dist = float('inf')
            closest = None
            for i, (rho, theta) in enumerate(self.lines):
                dist = abs(x * np.cos(theta) + y * np.sin(theta) - rho)
                if dist < min_dist:
                    min_dist = dist
                    closest = i
            if closest is not None and closest not in self.selected_lines and len(self.selected_lines) < 2:
                self.selected_lines.append(closest)
                self.update_display()

    def update_display(self):
        """Update display with polygons."""
        if self.img_coarsened is None:
            return
        self.graphics_view.clear_polygons()
        filter_type = self.filter_group.checkedButton().text()
        if filter_type == "Color":
            if self.selected_hsv is None:
                return
            full_mask = compute_similar_mask(
                self.img_hsv, self.selected_hsv, self.threshold_hue, self.threshold_sat
            )
            height, width = self.img_coarsened.shape[:2]
            if self.mask_points:
                mask_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(self.mask_points, fill=255)
                mask_array = np.array(mask_img) > 0
                full_mask &= mask_array
            if self.clicked_y is not None and self.clicked_x is not None:
                full_mask = select_boundary_component(full_mask, self.clicked_y, self.clicked_x)
            polygons = cluster_and_get_polygons(full_mask)
            for poly in polygons:
                points = poly.get_xy()
                self.graphics_view.add_polygon(points, QColor('red'))
        elif filter_type == "Mixed":
            if self.selected_hsv is None or self.selected_shape is None:
                return
            full_mask = compute_mixed_mask(
                self.img_hsv, self.img_coarsened, self.selected_hsv, self.selected_shape,
                self.threshold_hue, self.threshold_sat, self.threshold_shape
            )
            height, width = self.img_coarsened.shape[:2]
            if self.mask_points:
                mask_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(self.mask_points, fill=255)
                mask_array = np.array(mask_img) > 0
                full_mask &= mask_array
            if self.clicked_y is not None and self.clicked_x is not None:
                full_mask = select_boundary_component(full_mask, self.clicked_y, self.clicked_x)
            polygons = cluster_and_get_polygons(full_mask)
            for poly in polygons:
                points = poly.get_xy()
                self.graphics_view.add_polygon(points, QColor('red'))
        elif filter_type == "Line":
            self.draw_lines()
            if len(self.selected_lines) == 2:
                self.draw_line_polygon()

    def draw_line_polygon(self):
        """Draw polygon from selected lines."""
        idx1, idx2 = self.selected_lines
        rho1, theta1 = self.lines[idx1]
        rho2, theta2 = self.lines[idx2]
        height, width = self.img_coarsened.shape[:2]
        points1 = get_line_boundary_points(rho1, theta1, width, height)
        points2 = get_line_boundary_points(rho2, theta2, width, height)
        if len(points1) == 2 and len(points2) == 2:
            all_points = points1 + list(reversed(points2))
            self.graphics_view.add_polygon(all_points, QColor('red'))

    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.showNormal()
        elif event.key() == Qt.Key.Key_W and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.close()
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.graphics_view.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.graphics_view.zoom_out()
        super().keyPressEvent(event)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Image labeling GUI.")
    parser.add_argument("directory", nargs="?", help="Path to the image directory.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    dir_path = args.directory
    if not dir_path:
        dir_path = QFileDialog.getExistingDirectory(None, "Select image directory")
    if not dir_path:
        sys.exit(1)

    gui = ImageLabelerGUI(dir_path)
    sys.exit(app.exec())
