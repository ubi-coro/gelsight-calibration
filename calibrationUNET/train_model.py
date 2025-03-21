import json
import os
from dataclasses import dataclass
import logging
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ColorGradientDataset, GradientCNN

# Configure logging
logging.basicConfig(level=logging.INFO)

"""
This script trains the gradient prediction network.
The network is trained as MLP taking the pixel BGRXY as input and predict the gradients gx, gy.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are collected.
    - Collected tactile images are labeled.
    - Labeled data are prepared into dataset.

Usage:
    python train.py --calib_dir CALIB_DIR [--n_epochs N_EPOCHS] [--lr LR] [--device {cpu, cuda}]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --n_epochs: (Optional) Number of training epochs. Default is 200.
    --lr: (Optional) Learning rate. Default is 0.002.
    --device: (Optional) The device to train the network. Can choose between cpu and cuda. Default is cpu.
"""


@dataclass
class TrainConfig:
    """
    Configuration class for the training of the gradient prediction network.
    """
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dirs
    data_dir: str = "/media/local/data"  # directory to the calibration data
    base_dir: str = f"{data_dir}/training/{_timestamp}"
    model_dir: str = os.path.join(data_dir, "models")  # This path is intended for storing or locating training log files.

    device: str = "cuda"  # device to train the network. Must be "cuda" or "cpu". Default = "cuda"

    batch_size: int = 2  # batch size
    n_epochs: int = 50  # number of epochs. Default = 50
    lr: float = 0.002  # learning rate. Default = 0.002


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


def _load_data_dirs(data_dir) -> (List[str], List[str]):
    data_path = os.path.join(data_dir, "train_test_split.json")
    data = load_json(data_path)
    train_dirs = [os.path.join(data_dir, rel_dir) for rel_dir in data["train"]]
    test_dirs = [os.path.join(data_dir, rel_dir) for rel_dir in data["test"]]
    return train_dirs, test_dirs


def setup(data_dir):
    train_dirs, test_dirs = _load_data_dirs(data_dir)
    train_data = load_data(train_dirs)
    test_data = load_data(test_dirs)
    train_dataset = ColorGradientDataset(train_data["images"], train_data["gradient_maps"])
    val_dataset = ColorGradientDataset(test_data["images"], test_data["gradient_maps"])
    return train_dataset, val_dataset


def load_data(reldirs: List[str]) -> Dict[str, List[np.ndarray]]:
    data = {"images": [], "gradient_maps": []}
    for reldir in reldirs:
        path = os.path.join(reldir, "data.npz")
        if not os.path.isfile(path):
            raise ValueError(f"Data file {path} does not exist")
        loaded = np.load(path)
        data["gradient_maps"].append(loaded["gradient_map"])
        data["images"].append(loaded["image"])
    return data

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.device

        # Create the CNN Net for training
        self.net = GradientCNN().to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Create train and test Dataloader
        train_dataset, val_dataset = setup(self.config.data_dir)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        self.best_loss = np.inf

    def train_model(self):
        # Initial evaluation
        train_mae = evaluate(self.net, self.train_dataloader, self.device)
        test_mae = evaluate(self.net, self.val_dataloader, self.device)
        traj = {"train_maes": [train_mae], "test_maes": [test_mae]}
        print("without train, Train MAE: %.4f, Test MAE: %.4f" % (train_mae, test_mae))

        # Train the model
        for epoch_idx in range(self.config.n_epochs):
            train_mean_loss, val_mean_loss = self.epoch()
            traj["train_maes"].append(train_mean_loss)
            traj["test_maes"].append(val_mean_loss)
            print("Epoch %i, Train MAE: %.4f, Test MAE: %.4f"% (epoch_idx, train_mean_loss, val_mean_loss)            )
            self.scheduler.step()
            # Save model every 10 steps
            if (epoch_idx + 1) % 10 == 0:
                # Transfer weights to MLP Net and save
                save_path = os.path.join(self.config.model_dir, f"nnmodel_e{epoch_idx}.pth")
                torch.save(self.net.state_dict(), save_path)
        self.save_loss_plot(traj)

    def save_loss_plot(self, traj):
        # Save the training curve
        save_path = os.path.join(self.config.model_dir, "training_curve.png")
        plt.plot(np.arange(len(traj["train_maes"])), traj["train_maes"], color="blue")
        plt.plot(np.arange(len(traj["test_maes"])), traj["test_maes"], color="red")
        plt.xlabel("Epochs")
        plt.ylabel("MAE (rad)")
        plt.title("MAE Curve")
        plt.savefig(save_path)
        plt.close()

    def epoch(self):
        losses = []
        self.net.train()
        for image, gradient_map in tqdm(self.train_dataloader):
            loss = self.training_step(gradient_map, image)
            losses.append(loss)
        self.net.eval()
        train_mean_loss = np.mean(np.concatenate(losses))
        val_mean_loss = evaluate(self.net, self.val_dataloader, self.device)

        if val_mean_loss < self.best_loss:
            self.best_loss = val_mean_loss
            self.save_model("best_model")

        return train_mean_loss, val_mean_loss

    def training_step(self, gradient_map, image):
        image = image.to(self.device)
        gradient_map = gradient_map.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(image)
        loss = self.criterion(outputs, gradient_map)
        loss.backward()
        self.optimizer.step()
        diffs = outputs - gradient_map
        return np.abs(diffs.cpu().detach().numpy())

    def save_model(self, model_name):
        save_path = os.path.join(self.config.model_dir, f"{model_name}.pth")
        torch.save(self.net.state_dict(), save_path)

def evaluate(net, dataloader, device):
    """
    Evaluate the network loss on the dataset.

    :param net: nn.Module; the network to evaluate.
    :param dataloader: DataLoader; the dataloader for the dataset.
    :param device: str; the device to evaluate the network.
    """
    losses = []
    for image, gradient_map in dataloader:
        image = image.to(device)
        gradient_map = gradient_map.to(device)
        outputs = net(image)
        diffs = outputs - gradient_map
        losses.append(np.abs(diffs.cpu().detach().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


if __name__ == "__main__":
    config = TrainConfig()  # Instantiate configuration
    trainer = Trainer(config)  # Pass the config instance to the Trainer
    trainer.train_model()
