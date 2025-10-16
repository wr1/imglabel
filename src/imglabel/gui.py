"""Graphical user interface for imglabel using Tkinter."""

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw
import numpy as np
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
)


class ImageLabelerGUI:
    """Main GUI class for image labeling."""
    def __init__(self, root, dir_path=None):
        self.root = root
        self.root.title("imglabel GUI")
        self.selected_hsv = None
        self.selected_shape = None
        self.threshold_hue = 0.05
        self.threshold_sat = 0.2
        self.threshold_shape = 0.5
        self.img_coarsened = None
        self.img_hsv = None
        self.polygons = []
        self.photo = None
        self.dir_path = dir_path
        self.zoom_factor = 1.0
        self.lines = []
        self.selected_lines = []
        self.line_ids = []
        self.line_sensitivity = 1.5
        self.mask_points = None
        self.setup_ui()
        if self.dir_path:
            self.load_directory(self.dir_path)
            self.select_first_image()
        self.on_filter_change()  # Initialize visibility and display

    def setup_ui(self):
        """Set up the user interface components."""
        # Left frame for list and controls
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # File list
        self.file_list = tk.Listbox(self.left_frame, width=20)
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind('<<ListboxSelect>>', self.on_file_select)

        # Controls below the list
        self.control_frame = tk.Frame(self.left_frame)
        self.control_frame.pack(fill=tk.X)
        self.load_criteria_btn = tk.Button(
            self.control_frame, text="Load Criteria", command=self.load_criteria
        )
        self.load_criteria_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_criteria_btn = tk.Button(
            self.control_frame, text="Save Criteria", command=self.save_criteria
        )
        self.save_criteria_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Middle: Canvas for image
        self.canvas = tk.Canvas(self.root, bg='white')
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        # Zoom bindings
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.zoom_in)
        self.canvas.bind('<Button-5>', self.zoom_out)

        # Right: Filter and settings pane
        self.right_frame = tk.Frame(self.root, width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(self.right_frame, text="Filter:").pack(pady=5)
        self.filter_var = tk.StringVar(value="Color Similarity")
        tk.Radiobutton(
            self.right_frame, text="Color Similarity", variable=self.filter_var,
            value="Color Similarity", command=self.on_filter_change
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            self.right_frame, text="Mixed Filter", variable=self.filter_var,
            value="Mixed Filter", command=self.on_filter_change
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            self.right_frame, text="Line Filter", variable=self.filter_var,
            value="Line Filter", command=self.on_filter_change
        ).pack(anchor=tk.W)

        # Settings
        self.settings_frame = tk.Frame(self.right_frame)
        self.settings_frame.pack(pady=10)
        tk.Label(self.settings_frame, text="Settings:").pack()
        # Color Similarity and Mixed settings
        self.hue_label = tk.Label(self.settings_frame, text="Hue Threshold:")
        self.hue_entry = tk.Entry(self.settings_frame)
        self.hue_entry.insert(0, str(self.threshold_hue))
        self.hue_entry.bind('<Return>', lambda e: self.update_hue_threshold())
        self.hue_entry.bind('<FocusOut>', lambda e: self.update_hue_threshold())
        self.sat_label = tk.Label(self.settings_frame, text="Sat Threshold:")
        self.sat_entry = tk.Entry(self.settings_frame)
        self.sat_entry.insert(0, str(self.threshold_sat))
        self.sat_entry.bind('<Return>', lambda e: self.update_sat_threshold())
        self.sat_entry.bind('<FocusOut>', lambda e: self.update_sat_threshold())
        self.shape_label = tk.Label(self.settings_frame, text="Shape Threshold:")
        self.shape_entry = tk.Entry(self.settings_frame)
        self.shape_entry.insert(0, str(self.threshold_shape))
        self.shape_entry.bind('<Return>', lambda e: self.update_shape_threshold())
        self.shape_entry.bind('<FocusOut>', lambda e: self.update_shape_threshold())
        # Line Filter settings
        self.sensitivity_label = tk.Label(self.settings_frame, text="Sensitivity:")
        self.sensitivity_scale = tk.Scale(
            self.settings_frame, from_=0.5, to=3.0, resolution=0.1,
            orient=tk.HORIZONTAL, command=self.update_sensitivity
        )
        self.sensitivity_scale.set(self.line_sensitivity)
        self.mask_btn = tk.Button(self.settings_frame, text="MASK", command=self.set_mask)

    def on_filter_change(self):
        """Handle filter change."""
        self.selected_lines = []
        self.selected_hsv = None
        self.selected_shape = None
        filter_type = self.filter_var.get()
        # Hide all settings
        self.hue_label.pack_forget()
        self.hue_entry.pack_forget()
        self.sat_label.pack_forget()
        self.sat_entry.pack_forget()
        self.shape_label.pack_forget()
        self.shape_entry.pack_forget()
        self.sensitivity_label.pack_forget()
        self.sensitivity_scale.pack_forget()
        self.mask_btn.pack_forget()
        # Show relevant settings
        if filter_type in ("Color Similarity", "Mixed Filter"):
            self.hue_label.pack()
            self.hue_entry.pack()
            self.sat_label.pack()
            self.sat_entry.pack()
            if filter_type == "Mixed Filter":
                self.shape_label.pack()
                self.shape_entry.pack()
        elif filter_type == "Line Filter":
            self.sensitivity_label.pack()
            self.sensitivity_scale.pack()
            self.mask_btn.pack()
        self.display_image()

    def update_hue_threshold(self):
        """Update hue threshold from entry."""
        try:
            value = float(self.hue_entry.get())
            self.threshold_hue = value
            self.update_display()
        except ValueError:
            self.hue_entry.delete(0, tk.END)
            self.hue_entry.insert(0, str(self.threshold_hue))

    def update_sat_threshold(self):
        """Update saturation threshold from entry."""
        try:
            value = float(self.sat_entry.get())
            self.threshold_sat = value
            self.update_display()
        except ValueError:
            self.sat_entry.delete(0, tk.END)
            self.sat_entry.insert(0, str(self.threshold_sat))

    def update_shape_threshold(self):
        """Update shape threshold from entry."""
        try:
            value = float(self.shape_entry.get())
            self.threshold_shape = value
            self.update_display()
        except ValueError:
            self.shape_entry.delete(0, tk.END)
            self.shape_entry.insert(0, str(self.threshold_shape))

    def update_sensitivity(self, value):
        """Update sensitivity value and re-detect lines."""
        self.line_sensitivity = float(value)
        self.display_image()

    def set_mask(self):
        """Set the mask from the selected lines."""
        if len(self.selected_lines) == 2:
            idx1, idx2 = self.selected_lines
            rho1, theta1 = self.lines[idx1]
            rho2, theta2 = self.lines[idx2]
            height, width = self.img_coarsened.shape[:2]
            points1 = get_line_boundary_points(rho1, theta1, width, height)
            points2 = get_line_boundary_points(rho2, theta2, width, height)
            if len(points1) == 2 and len(points2) == 2:
                self.mask_points = points1 + list(reversed(points2))
                messagebox.showinfo("Mask Set", "Mask set from selected lines.")
        else:
            messagebox.showwarning("No Selection", "Select two lines first.")

    def load_directory(self, dir_path):
        """Load image files from the selected directory."""
        self.dir_path = dir_path
        files = [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.file_list.delete(0, tk.END)
        for f in files:
            self.file_list.insert(tk.END, f)

    def select_first_image(self):
        """Select and load the first image in the list."""
        if self.file_list.size() > 0:
            self.file_list.selection_set(0)
            self.on_file_select(None)

    def on_file_select(self, event):
        """Handle file selection from the list."""
        selection = self.file_list.curselection()
        if selection:
            filename = self.file_list.get(selection[0])
            self.load_image(os.path.join(self.dir_path, filename))

    def load_image(self, path):
        """Load and display the selected image."""
        try:
            self.img_coarsened, self.img_hsv = load_and_coarsen_image(path)
            # Reset per-image state
            self.selected_hsv = None
            self.selected_shape = None
            self.selected_lines = []
            self.mask_points = None
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def display_image(self):
        """Display the image on the canvas with current zoom."""
        if self.img_coarsened is None:
            return
        self.canvas.delete("all")
        img = Image.fromarray(self.img_coarsened)
        width, height = img.size
        zoomed_width = int(width * self.zoom_factor)
        zoomed_height = int(height * self.zoom_factor)
        img_zoomed = img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_zoomed)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))
        # Draw mask if set and in Color Similarity or Mixed
        if self.mask_points and self.filter_var.get() in ("Color Similarity", "Mixed Filter"):
            flat_points = [coord * self.zoom_factor for point in self.mask_points for coord in point]
            self.canvas.create_polygon(flat_points, fill='lightgray', outline='black', stipple='gray50')
        filter_type = self.filter_var.get()
        if filter_type == "Line Filter":
            img_gray = rgb2gray(self.img_coarsened)
            self.lines = detect_lines(img_gray, sigma=self.line_sensitivity)
            self.draw_lines()
        self.update_display()

    def draw_lines(self):
        """Draw detected lines on the canvas."""
        self.line_ids = []
        for i, (rho, theta) in enumerate(self.lines):
            points = get_line_boundary_points(rho, theta, self.img_coarsened.shape[1], self.img_coarsened.shape[0])
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x1z = x1 * self.zoom_factor
                y1z = y1 * self.zoom_factor
                x2z = x2 * self.zoom_factor
                y2z = y2 * self.zoom_factor
                color = 'red' if i in self.selected_lines else 'blue'
                line_id = self.canvas.create_line(x1z, y1z, x2z, y2z, fill=color, width=2)
                self.line_ids.append(line_id)

    def on_canvas_click(self, event):
        """Handle click on the canvas to select HSV patch or line."""
        if self.img_hsv is None:
            return
        x, y = event.x / self.zoom_factor, event.y / self.zoom_factor
        height, width = self.img_coarsened.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        filter_type = self.filter_var.get()
        if filter_type in ("Color Similarity", "Mixed Filter"):
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
            if filter_type == "Mixed Filter":
                patch_gray = rgb2gray(patch_rgb)
                edges = canny(patch_gray, sigma=1.0)
                self.selected_shape = np.mean(edges)
            self.update_display()
        elif filter_type == "Line Filter":
            # Find closest line
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
        """Update the display with highlighted polygons or selected lines."""
        if self.img_coarsened is None:
            return
        filter_type = self.filter_var.get()
        if filter_type == "Color Similarity":
            if self.selected_hsv is None:
                return
            full_mask = compute_similar_mask(
                self.img_hsv, self.selected_hsv, self.threshold_hue, self.threshold_sat
            )
            height, width = self.img_coarsened.shape[:2]
            if self.mask_points:
                # Create mask from polygon
                mask_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(self.mask_points, fill=255)
                mask_array = np.array(mask_img) > 0
                full_mask &= mask_array
            polygons = cluster_and_get_polygons(full_mask)
            # Clear previous polygons
            for poly_id in self.polygons:
                self.canvas.delete(poly_id)
            self.polygons = []
            for poly in polygons:
                points = poly.get_xy()
                flat_points = [coord * self.zoom_factor for point in points for coord in point]
                poly_id = self.canvas.create_polygon(
                    flat_points, outline='red', fill='', width=2
                )
                self.polygons.append(poly_id)
        elif filter_type == "Mixed Filter":
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
            polygons = cluster_and_get_polygons(full_mask)
            # Clear previous polygons
            for poly_id in self.polygons:
                self.canvas.delete(poly_id)
            self.polygons = []
            for poly in polygons:
                points = poly.get_xy()
                flat_points = [coord * self.zoom_factor for point in points for coord in point]
                poly_id = self.canvas.create_polygon(
                    flat_points, outline='red', fill='', width=2
                )
                self.polygons.append(poly_id)
        elif filter_type == "Line Filter":
            self.draw_lines()
            if len(self.selected_lines) == 2:
                self.draw_line_polygon()

    def draw_line_polygon(self):
        """Draw the polygon formed by the two selected lines."""
        idx1, idx2 = self.selected_lines
        rho1, theta1 = self.lines[idx1]
        rho2, theta2 = self.lines[idx2]
        height, width = self.img_coarsened.shape[:2]
        points1 = get_line_boundary_points(rho1, theta1, width, height)
        points2 = get_line_boundary_points(rho2, theta2, width, height)
        if len(points1) == 2 and len(points2) == 2:
            # Order points to form a polygon without crossing
            all_points = points1 + list(reversed(points2))
            flat_points = [coord * self.zoom_factor for point in all_points for coord in point]
            self.canvas.create_polygon(flat_points, fill='red', outline='red', stipple='gray50')

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming (Windows)."""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self, event=None):
        """Zoom in on the image."""
        self.zoom_factor *= 1.1
        self.zoom_factor = min(5.0, self.zoom_factor)
        self.display_image()

    def zoom_out(self, event=None):
        """Zoom out on the image."""
        self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, self.zoom_factor)
        self.display_image()

    def load_criteria(self):
        """Load criteria from a YAML file."""
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if file_path:
            try:
                (
                    self.selected_hsv,
                    self.threshold_hue,
                    self.threshold_sat,
                ) = load_criteria(file_path)
                self.hue_entry.delete(0, tk.END)
                self.hue_entry.insert(0, str(self.threshold_hue))
                self.sat_entry.delete(0, tk.END)
                self.sat_entry.insert(0, str(self.threshold_sat))
                self.update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load criteria: {e}")

    def save_criteria(self):
        """Save current criteria to a YAML file."""
        if self.selected_hsv is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")]
            )
            if file_path:
                save_criteria(
                    self.selected_hsv, self.threshold_hue, self.threshold_sat, file_path
                )
                messagebox.showinfo("Saved", f"Criteria saved to {file_path}")
        else:
            messagebox.showwarning("No Selection", "Please select a patch first.")


def main():
    """Main entry point for the GUI."""
    parser = argparse.ArgumentParser(description="Image labeling GUI.")
    parser.add_argument("directory", nargs="?", help="Path to the image directory.")
    args = parser.parse_args()

    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
    root.bind('<Control-w>', lambda e: root.quit())
    root.bind('<Control-plus>', lambda e: gui.zoom_in())
    root.bind('<Control-minus>', lambda e: gui.zoom_out())
    root.bind('<Control-equal>', lambda e: gui.zoom_in())  # For some keyboards

    dir_path = args.directory
    if not dir_path:
        dir_path = filedialog.askdirectory(title="Select image directory")
    if not dir_path:
        sys.exit(1)

    gui = ImageLabelerGUI(root, dir_path)
    root.mainloop()
