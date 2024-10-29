import argparse
import os

import cv2
import yaml

from gs_sdk.gs_device import Camera

"""
This script demonstrates how to use the Camera class from the gs_sdk package.

It loads a configuration file, initializes the Camera, and streaming images.

Usage:
    python stream_device.py --device_name {gsmini, digit}

Arguments:
    --device_name: The name of the device to stream from. Options are 'gsmini' or 'digit'.
                   Default is 'gsmini'.

Press any key to quit the streaming session.
"""

config_dir = os.path.join(os.path.dirname(__file__), "configs")


def stream_device():
    # Argument parser
    parser = argparse.ArgumentParser(description="Read and show image from the device.")
    parser.add_argument(
        "-n",
        "--device_name",
        type=str,
        choices=["gsmini", "digit"],
        default="gsmini",
        help="The name of the device",
    )
    args = parser.parse_args()

    # Load the device configuration
    if args.device_name == "gsmini":
        config_file = "gsmini.yaml"
    elif args.device_name == "digit":
        config_file = "digit.yaml"
    else:
        raise ValueError("Unknown device name %s." % args.device_name)
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        imgh = config["imgh"]
        imgw = config["imgw"]

    # Create device and stream the device
    device = Camera(device_name, imgh, imgw)
    device.connect()
    print("\nPrss any key to quit.\n")
    while True:
        image = device.get_image()
        cv2.imshow(device_name, image)
        key = cv2.waitKey(1)
        if key != -1:
            break
    device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_device()
