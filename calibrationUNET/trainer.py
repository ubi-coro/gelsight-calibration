import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import lightning as L

from models import GradientCNN, ColorGradientDataset
import draccus

# Configure logging
logging.basicConfig(level=logging.INFO)


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


class GradientModel(L.LightningModule):
    """PyTorch Lightning module for the GradientCNN model."""

    def __init__(self, lr: float = 0.002):
        super().__init__()
        self.model = GradientCNN()
        self.criterion = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, gradient_maps = batch
        outputs = self(images)
        loss = self.criterion(outputs, gradient_maps)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gradient_maps = batch
        outputs = self(images)
        loss = self.criterion(outputs, gradient_maps)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]


class GradientDataModule(L.LightningDataModule):
    """Data module for handling dataset loading and preparation."""

    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_dirs, test_dirs = self._load_data_dirs()
        train_data = self.load_data(train_dirs)
        test_data = self.load_data(test_dirs)
        self.train_dataset = ColorGradientDataset(train_data["images"], train_data["gradient_maps"])
        self.val_dataset = ColorGradientDataset(test_data["images"], test_data["gradient_maps"])

    def _load_data_dirs(self) -> (List[str], List[str]):
        data_path = os.path.join(self.data_dir, "train_test_split.json")
        data = load_json(data_path)
        train_dirs = [os.path.join(self.data_dir, rel_dir) for rel_dir in data["train"]]
        test_dirs = [os.path.join(self.data_dir, rel_dir) for rel_dir in data["test"]]
        return train_dirs, test_dirs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def load_data(self, reldirs: List[str]) -> Dict[str, List[np.ndarray]]:
        data = {"images": [], "gradient_maps": []}
        for reldir in reldirs:
            path = os.path.join(reldir, "data.npz")
            if not os.path.isfile(path):
                raise ValueError(f"Data file {path} does not exist")
            loaded = np.load(path)
            data["gradient_maps"].append(loaded["gradient_map"])
            data["images"].append(loaded["image"])
        return data


@dataclass
class Config:
    """Configuration schema for experiment parameters."""
    data_dir: str = "/media/local/data"  # Default data directory
    batch_size: int = 2  # Default batch size
    epochs: int = 50  # Number of training epochs
    learning_rate: float = 0.002  # Learning rate


@draccus.wrap()
def main(cfg: Config):
    """Main function to run the training."""
    data_module = GradientDataModule(cfg.data_dir, cfg.batch_size)
    model = GradientModel(cfg.learning_rate)

    trainer = L.Trainer(max_epochs=cfg.epochs, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
