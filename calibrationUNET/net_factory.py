from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Type

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from calibrationUNET.models import GradientCNN, ColorGradientDataset
from calibration.models import BGRXYMLPNet_, BGRXYDataset

CREATOR_REGISTRY: Dict[str, Type[NetDataset]] = {}


def register_creator(name):
    """Decorator to register a new creator."""

    def wrapper(creator_cls):
        CREATOR_REGISTRY[name] = creator_cls
        return creator_cls

    return wrapper


def get_creator(net_name: str) -> NetDataset:
    """Get the creator based on the registered name."""
    creator_cls = CREATOR_REGISTRY.get(net_name)
    if creator_cls is None:
        raise ValueError(f"Unknown net name: {net_name}")
    return creator_cls()


class NetDataset(ABC):
    def __init__(self):
        self._net: nn.Module | None = None

    def get_net(self) -> nn.Module:
        return self._net


    @staticmethod
    def load_json(file_path: str) -> dict:
        """Load JSON file and handle exceptions."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {file_path}")
            raise e

    def _load_data_dirs(self, data_dir) -> (List[str], List[str]):
        json_path = os.path.join(data_dir, "train_test_split.json")
        data = self.load_json(json_path)
        train_dirs = [os.path.join(data_dir, rel_dir) for rel_dir in data["train"]]
        test_dirs = [os.path.join(data_dir, rel_dir) for rel_dir in data["test"]]
        return train_dirs, test_dirs


    @abstractmethod
    def setup(self, data_dir) -> (Dataset, Dataset):
        pass


@register_creator("GradientCNN")
class CNNCreator(NetDataset, ABC):
    def __init__(self):
        super().__init__()
        self._net = GradientCNN()

    def setup(self, data_dir) -> (Dataset, Dataset):
        train_dirs, test_dirs = self._load_data_dirs(data_dir)
        train_data = self._load_data(train_dirs)
        test_data = self._load_data(test_dirs)
        train_dataset = ColorGradientDataset(train_data["images"], train_data["gradient_maps"])
        val_dataset = ColorGradientDataset(test_data["images"], test_data["gradient_maps"])
        return train_dataset, val_dataset

    @staticmethod
    def _load_data(rel_dirs: List[str]) -> Dict[str, List[np.ndarray]]:
        data = {"images": [], "gradient_maps": []}
        for rel_dir in rel_dirs:
            path = os.path.join(rel_dir, "data.npz")
            if not os.path.isfile(path):
                raise ValueError(f"Data file {path} does not exist")
            loaded = np.load(path)
            data["gradient_maps"].append(loaded["gradient_map"])
            data["images"].append(loaded["image"])
        return data


@register_creator("PixelNet")
class PixelNetCreator(NetDataset, ABC):
    def __init__(self):
        super().__init__()
        self._net = BGRXYMLPNet_()

    @staticmethod
    def _load_data(rel_dirs: List[str]) -> Dict[str, List[np.ndarray]]:
        data = {"all_bgrxys": [], "all_gxyangles": []}
        for rel_dir in rel_dirs:
            path = os.path.join(rel_dir, "data.npz")
            if not os.path.isfile(path):
                raise ValueError(f"Data file {path} does not exist")
            loaded = np.load(path)
            mask = loaded["mask"]
            data["all_bgrxys"].append(loaded["bgrxys"][mask])
            data["all_gxyangles"].append(loaded["gxyangles"][mask])
        return data

    def setup(self, data_dir) -> (Dataset, Dataset):
        train_dirs, test_dirs = self._load_data_dirs(data_dir)
        train_data = self._load_data(train_dirs)
        test_data = self._load_data(test_dirs)

        # Load background data
        bg_path = os.path.join(data_dir, "background_data.npz")
        bg_data = np.load(bg_path)
        mask = np.full_like(bg_data["mask"], True, dtype=bool)
        bgrxys = bg_data["bgrxys"][mask]
        gxyangles = bg_data["gxyangles"][mask]
        perm = np.random.permutation(len(bgrxys))
        n_train = np.sum([len(bgrxys) for bgrxys in train_data["all_bgrxys"]]) // 5
        n_test = np.sum([len(bgrxys) for bgrxys in test_data["all_bgrxys"]]) // 5
        if n_train + n_test > len(bgrxys):
            n_train = 4 * len(bgrxys) // 5
            n_test = len(bgrxys) // 5
        train_data["all_bgrxys"].append(bgrxys[perm[:n_train]])
        train_data["all_gxyangles"].append(gxyangles[perm[:n_train]])
        test_data["all_bgrxys"].append(bgrxys[perm[n_train: n_train + n_test]])
        test_data["all_gxyangles"].append(gxyangles[perm[n_train: n_train + n_test]])
        # Construct the train and test dataset
        train_bgrxys = np.concatenate(train_data["all_bgrxys"])
        train_gxyangles = np.concatenate(train_data["all_gxyangles"])
        test_bgrxys = np.concatenate(test_data["all_bgrxys"])
        test_gxyangles = np.concatenate(test_data["all_gxyangles"])

        train_dataset = BGRXYDataset(train_bgrxys, train_gxyangles)
        val_dataset = BGRXYDataset(test_bgrxys, test_gxyangles)
        return train_dataset, val_dataset