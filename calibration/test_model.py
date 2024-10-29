import argparse
import os

import cv2
import numpy as np
import yaml

from gs_sdk.gs_device import Camera
from gs_sdk.gs_reconstruct import Reconstructor

"""
This script tests the calibrated model real-time reconstructing local patches with the sensor.

Prerequisite:
    - Collect data and train the calibration model.
Instructions:
    - Connect the sensor to the computer.
    - Run this script, wait a bit for background collection, press any key to quit the streaming session.

Usage:
    python test.py --calib_dir CALIB_DIR [--config_path CONFIG_PATH]

Arguments:
    --calib_dir: Path to the directory where the collected data is stored.
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                   If not provided, GelSight Mini is assumed.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


def test_model():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Read image from the device and reconstruct based on the calibrated model."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="place where the calibration data is stored",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the sensor information",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    args = parser.parse_args()

    # Load the device configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        ppmm = config["ppmm"]

    # Create device and the reconstructor
    device = Camera(device_name, imgh, imgw)
    device.connect()
    model_path = os.path.join(args.calib_dir, "model", "nnmodel.pth")
    recon = Reconstructor(model_path, device="cpu")

    # Collect background images
    print("Collecting 10 background images, please wait ...")
    bg_images = []
    for _ in range(10):
        image = device.get_image()
        bg_images.append(image)
    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
    recon.load_bg(bg_image)

    # Real-time reconstruct
    print("\nPrss any key to quit.\n")
    while True:
        image = device.get_image()
        G, H, C = recon.get_surface_info(image, ppmm)
        # Create the image for gradient visualization
        red = G[:, :, 0] * 255 / 3.0 + 127
        red = np.clip(red, 0, 255)
        blue = G[:, :, 1] * 255 / 3.0 + 127
        blue = np.clip(blue, 0, 255)
        grad_image = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
        # Display
        cv2.imshow(device_name, grad_image)
        key = cv2.waitKey(1)
        if key != -1:
            break

    device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_model()
