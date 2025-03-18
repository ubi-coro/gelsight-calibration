import gc
import os
import argparse
from time import sleep

import cv2
import numpy as np
import nanogui as ng
from nanogui import Texture
from nanogui import glfw
import yaml

from calibration.utils import load_csv_as_dict

"""
Rewrite from Zilin Si's code: https://github.com/Robo-Touch/Taxim

This script is for manually labeling the contact circle in the tactile image for sensor calibration.
The labeled data will save the center and radius of the circle in the tactile image.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are collected.
Instruction: 
    1. Runs the script, 
    2. Mouse-press 'Open' to select the directory where the collected data are saved.
    3. Press left/right/up/down to control the circle's location,
       Press m/p to decrease/increase the circle's radius,
       Press f/c to decrease/increase the circle's moving step.
       Mouse-press 'Calibrate' to save the labeled data.
    4. Repeat step 3 for all the tactile images and close Nanogui when done.

Usage:
    python label_data.py --calib_dir CALIB_DIR [--config_path CONFIG_PATH] [--display_difference] [--detect_circle]

Arguments:
    --calib_dir: Path to the directory where the collected data are saved
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                   If not provided, GelSight Mini is assumed.
    --display_difference: (Store True) Display the difference between the background image.
    --detect_circle: (Store True) Automatically detect the circle in the image.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


class Circle:
    """the circle drawed on the tactile image to get the contact size"""

    color_red = (128, 0, 0)
    color_black = (0, 0, 0)
    opacity = 0.8

    def __init__(self, x, y, radius=200, increments_rough_estimation=15, increments_fine_tuning=1):
        self.center = [x, y]
        self.radius = radius
        self.increments_rough_estimation = increments_rough_estimation
        self.increments_fine_tuning = increments_fine_tuning

    def get_increments(self, stage: int):
        if stage == 0:
            return self.increments_rough_estimation
        else:
            return self.increments_fine_tuning

    def get_color_circle(self, stage: int):
        if stage == 0:
            return self.color_red
        else:
            return self.color_black


class CalibrateApp(ng.Screen):
    fnames = list()
    read_all = False  # flag to indicate if all images have been read
    load_img = True
    change = False
    scale_factor = 3

    def __init__(
            self, calib_data, imgw, imgh, display_difference=False, detect_circle=False
    ):
        super(CalibrateApp, self).__init__((int(imgw*1.1), int(imgh*1.1)), "Gelsight Calibration App")
        self.imgw = imgw
        self.imgh = imgh
        self.display_difference = display_difference
        self.detect_circle = detect_circle
        # Load background
        self.bg_img = cv2.imread(os.path.join(calib_data, "background.png"))
        # Initialize the circle
        self.circle = Circle(self.imgw / 2, self.imgh / 2)
        self.stage = 0

        def open_cb():
            self.parent_dir = calib_data
            # Read the catalog and create a list of all filenames
            catalog_dict = load_csv_as_dict(
                os.path.join(self.parent_dir, "catalog.csv")
            )
            self.fnames = [
                os.path.join(self.parent_dir, fname)
                for fname in catalog_dict["experiment_reldir"]
            ]
            self.circle_radius = [
                float(radius) for radius in catalog_dict["diameter(mm)"]
            ]
            print(
                f"Selected directory = {self.parent_dir}, total {len(self.fnames)} images"
            )
            self.img_idx = 0

        open_cb()

        ###########
        self.img_view = ng.ImageView(self)
        self.img_tex = ng.Texture(
            pixel_format=Texture.PixelFormat.RGB,
            component_format=Texture.ComponentFormat.UInt8,
            size=[imgw, imgh],
            min_interpolation_mode=Texture.InterpolationMode.Trilinear,
            mag_interpolation_mode=Texture.InterpolationMode.Nearest,
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget,
        )
        self.perform_layout()

    def calibrate_cb(self):
        frame = self.orig_img
        center = self.circle.center
        radius = self.circle.radius
        print(f"Frame {self.img_idx}: center = {center}, radius = is {radius}")
        # save the data for each individual frame instead of creating a long list and save it at the end
        # save the radius and center to npz file
        save_dir = os.path.join(self.fnames[self.img_idx], "label.npz")
        np.savez(save_dir, center=center, radius=radius)
        # save the labeled image
        labeled_img = self.overlay_circle(frame, self.circle)
        labeled_img_path = os.path.join(self.fnames[self.img_idx], "labeled.png")
        cv2.imwrite(labeled_img_path, labeled_img)

        # Update img index
        self.load_img = True
        self.update_img_idx()
        self.reset_zoom()

    def update_img_idx(self):
        self.img_idx += 1
        if self.img_idx == len(self.fnames) - 1:
            self.read_all = True

    def overlay_circle(self, orig_img, circle):
        center = circle.center
        radius = circle.radius
        color_circle = circle.get_color_circle(self.stage)
        opacity = circle.opacity

        overlay = orig_img.copy()
        center_tuple = (int(center[0]), int(center[1]))
        cv2.circle(overlay, center_tuple, radius, color_circle, 1)
        cv2.circle(overlay, center_tuple, 1, color_circle, -1)
        cv2.addWeighted(overlay, opacity, orig_img, 1 - opacity, 0, overlay)
        return overlay

    def draw(self, ctx):
        # self.img_window.set_size((int(self.imgw*1.05), int(self.imgh*1.05)))

        self.img_view.set_size((int(0.9*self.imgw), int(0.9*self.imgh)))

        # load a new image
        if self.load_img and len(self.fnames) > 0 and not self.read_all:
            print("Loading %s" % self.fnames[self.img_idx])

            # Load img
            self.orig_img = cv2.imread(
                os.path.join(self.fnames[self.img_idx], "gelsight.png")
            )
            # Initialize the circle pose
            if self.detect_circle:
                diff_image = self.orig_img.astype(np.float32) - self.bg_img.astype(
                    np.float32
                )
                color_mask = np.linalg.norm(diff_image, axis=-1) > 15
                color_mask = cv2.dilate(
                    color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
                )
                color_mask = cv2.erode(
                    color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
                )
                contours, _ = cv2.findContours(
                    color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = self.circle.center[0]
                cy = self.circle.center[1]
            radius = max(self.circle.radius - 13, 5)
            radius = self.circle.radius
            self.circle = Circle(cx, cy, radius=radius)

        # Add circle and add img to viewer
        if (self.load_img and len(self.fnames) > 0) or self.change:
            self.load_img = False
            self.change = False
            # Add circle
            if self.display_difference:
                diff_img = (
                                   self.orig_img.astype(np.float32) - self.bg_img.astype(np.float32)
                           ) * 3
                diff_img = np.clip(diff_img, -127, 128) + np.ones_like(diff_img) * 127
                display_img = cv2.cvtColor(diff_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                display_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
            img = self.overlay_circle(display_img, self.circle)

            if self.img_tex.channels() > 3:
                height, width = img.shape[:2]
                alpha = 255 * np.ones((height, width, 1), dtype=img.dtype)
                img = np.concatenate((img, alpha), axis=2)

            # Add to img view
            self.img_tex.upload(img)
            self.img_view.set_image(self.img_tex)

        super(CalibrateApp, self).draw(ctx)

    def zoom(self, factor, center=None):
        """Zoom in or out by a given factor while moving the zoom center towards the middle of the image."""
        if center is None:
            # Default to the center of the image view
            center = (self.img_view.fixed_size()[0] // 2, self.img_view.fixed_size()[1] // 2)

        # Get the current scale and offset
        old_scale = self.img_view.scale()
        old_offset = self.img_view.offset()
        img_size = (self.imgw, self.imgh) # Get the size of the loaded image

        # Calculate the image's center point in screen coordinates
        image_center_x = (img_size[0] * old_scale) / 2
        image_center_y = (img_size[1] * old_scale) / 2

        # Convert screen coordinates to image coordinates
        img_center_x = (center[0] - old_offset[0]) / old_scale
        img_center_y = (center[1] - old_offset[1]) / old_scale

        # Apply the new zoom factor
        new_scale = old_scale * factor
        self.img_view.set_scale(new_scale)

        # Move the center point closer to the middle of the image
        new_offset_x = image_center_x - img_center_x * new_scale
        new_offset_y = image_center_y - img_center_y * new_scale

        self.img_view.set_offset((new_offset_x, new_offset_y))

    def reset_zoom(self):
        """Reset zoom to the original scale and center the image."""
        self.img_view.set_scale(1.0)  # Reset to the default scale (no zoom)

        # Get image and view sizes
        img_size = (self.imgw, self.imgh)
        view_size = (self.img_view.width(), self.img_view.height())

        # Calculate the offset to center the image
        offset_x = (view_size[0] - img_size[0]) / 2
        offset_y = (view_size[1] - img_size[1]) / 2

        # Apply the centered offset
        self.img_view.set_offset((offset_x, offset_y))

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(CalibrateApp, self).keyboard_event(key, scancode, action, modifiers):
            return True
        if action == glfw.RELEASE:
            return False
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True
        elif key == glfw.KEY_C:
            self.circle.increments_rough_estimation *= 2
        elif key == glfw.KEY_F:
            self.circle.increments_rough_estimation /= 2
        else:
            self.change = True
            if key == glfw.KEY_LEFT and (modifiers & glfw.MOD_CONTROL):
                self.circle.radius -= self.circle.get_increments(self.stage)
            elif key == glfw.KEY_RIGHT and (modifiers & glfw.MOD_CONTROL):
                self.circle.radius += self.circle.get_increments(self.stage)
            elif key == glfw.KEY_LEFT:
                self.circle.center[0] -= self.circle.get_increments(self.stage)
            elif key == glfw.KEY_RIGHT:
                self.circle.center[0] += self.circle.get_increments(self.stage)
            elif key == glfw.KEY_UP:
                self.circle.center[1] -= self.circle.get_increments(self.stage)
            elif key == glfw.KEY_DOWN:
                self.circle.center[1] += self.circle.get_increments(self.stage)
            elif key == glfw.KEY_BACKSPACE:
                self.back()
            elif key == glfw.KEY_ENTER or key == glfw.KEY_SPACE:
                self.finishe_stage()

        return False

    def load_previous(self):
        if self.img_idx == 0:
            return
        self.load_img = True
        self.img_idx -= 1

        save_dir = os.path.join(self.fnames[self.img_idx], "label.npz")
        data = np.load(save_dir)
        self.circle = Circle(*data["center"], data["radius"])


    def back(self):
        if self.stage == 1:
            self.stage = 0
            self.reset_zoom()
        else:
            self.load_previous()


    def finishe_stage(self):
        if self.stage == 0:
            self.stage = 1
            self.zoom(self.scale_factor, self.circle.center)
        else:
            self.stage = 0
            self.calibrate_cb()


def label_data():
    args = parse_args()

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["imgh"]
        imgw = config["imgw"]

    # Start the label process
    ng.init()
    app = CalibrateApp(
        args.calib_dir,
        imgw,
        imgh,
        display_difference=args.display_difference,
        detect_circle=args.detect_circle,
    )
    app.draw_all()
    app.set_visible(True)
    ng.mainloop(refresh=1 / 60.0 * 1000)
    del app
    gc.collect()
    ng.shutdown()


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Label the ball indenter data using Nanogui."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default=os.path.join(config_dir, "gsmini_highres.yaml"),
    )
    parser.add_argument(
        "-d",
        "--display_difference",
        action="store_true",
        help="Display the difference between the background image",
    )
    parser.add_argument(
        "-r",
        "--detect_circle",
        action="store_true",
        help="Automatically detect the circle in the image",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    label_data()
