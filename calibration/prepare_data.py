import argparse
import json
import os

import cv2
import numpy as np
import yaml

from calibration.utils import load_csv_as_dict
from gs_sdk.gs_reconstruct import image2bgrxys

"""
This script prepares dataset for the tactile sensor calibration.
It is based on the collected and labeled data.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are collected.
    - Collected tactile images are labeled.

Usage:
    python prepare_data.py --calib_dir CALIB_DIR [--config_path CONFIG_PATH] [--radius_reduction RADIUS_REDUCTION]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                   If not provided, GelSight Mini is assumed.
    --radius_reduction: (Optional) Reduce the radius of the labeled circle. This helps guarantee all labeled pixels are indented.
                        If not provided, 4 pixels will be reduced.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


def prepare_data():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Use the labeled collected data to prepare the dataset files (npz)."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path of the calibration data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    parser.add_argument(
        "-r",
        "--radius_reduction",
        type=float,
        help="reduce the radius of the labeled circle. When not considering shadows, this helps guarantee all labeled pixels are indented. ",
        default=4.0,
    )
    args = parser.parse_args()

    # Load the data_dict
    calib_dir = args.calib_dir
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    experiment_reldirs = np.array(data_dict["experiment_reldir"])

    # Split data into train and test and save the split information
    perm = np.random.permutation(len(experiment_reldirs))
    n_train = 4 * len(experiment_reldirs) // 5
    data_path = os.path.join(calib_dir, "train_test_split.json")
    dict_to_save = {
        "train": experiment_reldirs[perm[:n_train]].tolist(),
        "test": experiment_reldirs[perm[n_train:]].tolist(),
    }
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    # Extract the pixel data from each tactile image and calculate the gradients
    for experiment_reldir, diameter in zip(experiment_reldirs, diameters):
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        # Filter the non-indented pixels
        label_path = os.path.join(experiment_dir, "label.npz")
        label_data = np.load(label_path)
        center = label_data["center"]
        radius = label_data["radius"] - args.radius_reduction
        xys = np.dstack(
            np.meshgrid(
                np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy"
            )
        )
        dists = np.linalg.norm(xys - center, axis=2)
        mask = dists < radius

        # Find the gradient angles, prepare the data, and save the data
        ball_radius = diameter / ppmm / 2.0
        if ball_radius < radius:
            print(experiment_reldir)
            print("Press too deep, deeper than the ball radius")
            continue
        dxys = xys - center
        dists[np.logical_not(mask)] = 0.0
        dzs = np.sqrt(ball_radius**2 - np.square(dists))
        gxangles = np.arctan2(dxys[:, :, 0], dzs)
        gyangles = np.arctan2(dxys[:, :, 1], dzs)
        gxyangles = np.stack([gxangles, gyangles], axis=-1)
        gxyangles[np.logical_not(mask)] = np.array([0.0, 0.0])
        bgrxys = image2bgrxys(image)
        save_path = os.path.join(experiment_dir, "data.npz")
        np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)

    # Save the background data
    bg_path = os.path.join(calib_dir, "background.png")
    bg_image = cv2.imread(bg_path)
    bgrxys = image2bgrxys(bg_image)
    gxyangles = np.zeros((bg_image.shape[0], bg_image.shape[1], 2))
    mask = np.ones((bg_image.shape[0], bg_image.shape[1]), dtype=np.bool_)
    save_path = os.path.join(calib_dir, "background_data.npz")
    np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)


if __name__ == "__main__":
    prepare_data()
