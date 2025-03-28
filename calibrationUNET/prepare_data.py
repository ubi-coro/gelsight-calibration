import argparse
import json
import os
from dataclasses import dataclass

import cv2
import draccus
import numpy as np
import yaml
from tqdm import tqdm

from calibration.utils import load_csv_as_dict
from gs_sdk.gs_reconstruct import image2bgrxys

"""
This script prepares dataset for the tactile sensor calibration.
It is based on the collected and labeled data.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are collected.
    - Collected tactile images are labeled.

Usage:
    python prepare_data.py --calib_dir CALIB_DIR [--config_path CONFIG_PATH]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --device_config_path: (Optional) Path to the configuration file about the sensor dimensions.
                          If not provided, GelSight Mini is assumed.
"""

@dataclass
class PrepareConfig:
    device_config_path: str = "examples/configs/gsmini_highres.yaml"
    calib_dir: str = "/media/local/data"


def prepare_data():
    # Load the data_dict
    calib_dir, diameters, experiment_reldirs, ppmm = setup_data()

    # Extract the pixel data from each tactile image and calculate the gradients
    for experiment_reldir, diameter in tqdm(zip(experiment_reldirs, diameters), total=len(experiment_reldirs)):
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        prepare_image(diameter, experiment_dir, ppmm)

@draccus.wrap()
def setup_data(config: PrepareConfig):
    calib_dir = config.calib_dir
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    experiment_reldirs = np.array(data_dict["experiment_reldir"])
    # Split data into train and test and save the split information
    perm = np.random.permutation(len(experiment_reldirs))
    n_train = 4 * len(experiment_reldirs) // 5  # train: 80% test: 20%
    data_path = os.path.join(calib_dir, "train_test_split.json")
    dict_to_save = {
        "train": experiment_reldirs[perm[:n_train]].tolist(),
        "test": experiment_reldirs[perm[n_train:]].tolist(),
    }
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)
    with open(config.device_config_path, "r") as f:
        device_config = yaml.safe_load(f)
        ppmm = device_config["ppmm"]
    return calib_dir, diameters, experiment_reldirs, ppmm

def prepare_image(diameter, experiment_dir, ppmm):
    image_path = os.path.join(experiment_dir, "gelsight.png")
    image = cv2.imread(image_path)

    label_path = os.path.join(experiment_dir, "label.npz")
    label_data = np.load(label_path)
    center = label_data["center"]
    radius = label_data["radius"]
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
        print(experiment_dir)
        print("Press too deep, deeper than the ball radius")
        return
    dxys = xys - center
    dists[np.logical_not(mask)] = 0.0
    dzs = np.sqrt(ball_radius ** 2 - np.square(dists))
    gxangles = np.arctan2(dxys[:, :, 0], dzs)
    gyangles = np.arctan2(dxys[:, :, 1], dzs)
    gradient_map = np.stack([gxangles, gyangles], axis=-1)
    gradient_map[np.logical_not(mask)] = np.array([0.0, 0.0])
    save_path = os.path.join(experiment_dir, "data.npz")
    np.savez(save_path, gradient_map=gradient_map, image=image)


if __name__ == "__main__":
    prepare_data()
