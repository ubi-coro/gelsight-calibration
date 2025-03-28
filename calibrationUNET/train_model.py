import json
import os
from dataclasses import dataclass
import logging
from datetime import datetime
from typing import List, Dict

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from calibrationUNET.net_factory import get_creator

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
This script trains the gradient prediction network using a CNN-based architecture.
The network is trained to take tactile images as input and predict the gradients gx, gy.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are available.
    - Collected tactile images are labeled with corresponding gradient maps.
    - Labeled data is prepared into a compatible dataset format.

Usage:
    python train_model.py --data_dir DATA_DIR [--n_epochs N_EPOCHS] [--lr LR] [--device {cpu, cuda}]

Arguments:
    --data_dir: Path to the directory where the dataset is stored.
    --n_epochs: (Optional) Number of training epochs. Default is 50.
    --lr: (Optional) Learning rate. Default is 0.002.
    --device: (Optional) The device to train the network. Can choose between cpu and cuda. Default is cuda.
"""

@dataclass
class TrainConfig:
    """
    Configuration class for the training of the gradient prediction network.
    """
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dirs
    net_name: str = "PixelNet"
    data_dir: str = "/media/local/data"  # directory to the calibration data
    base_dir: str = os.path.join(data_dir, "training", f"{_timestamp}_{net_name}")
    model_dir: str = None  # This path is intended for storing or locating training log files.
    device: str = "cuda"  # device to train the network. Must be "cuda" or "cpu". Default = "cuda"

    batch_size: int = 1028*512  # batch size
    n_epochs: int = 50  # number of epochs. Default = 50
    lr: float = 0.002  # learning rate. Default = 0.002

    def __post_init__(self):
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        if self.model_dir is None:
            self.model_dir = os.path.join(self.base_dir, "model")
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

        logging.info("Setup creator")
        net_dataset_creator = get_creator(self.config.net_name)

        self.device = config.device

        logging.info("Create Net")
        self.net: nn.Module = net_dataset_creator.get_net().to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Create train and test Dataloader
        logging.info("Create Dataloader")
        train_dataset, val_dataset = net_dataset_creator.setup(self.config.data_dir)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        self.best_loss = np.inf

        logging.info("Finished setup")

    def train_model(self):
        logging.info("Start training")
        # Initial evaluation
        train_mae = evaluate(self.net, self.train_dataloader, self.device)
        test_mae = evaluate(self.net, self.val_dataloader, self.device)
        traj = {"train_maes": [train_mae], "test_maes": [test_mae]}
        logging.info("without train, Train MAE: %.4f, Test MAE: %.4f" % (train_mae, test_mae))

        # Train the model
        for epoch_idx in range(self.config.n_epochs):
            train_mean_loss, val_mean_loss = self.epoch()
            traj["train_maes"].append(train_mean_loss)
            traj["test_maes"].append(val_mean_loss)
            logging.info("Epoch %i, Train MAE: %.4f, Test MAE: %.4f" % (epoch_idx, train_mean_loss, val_mean_loss))
            self.scheduler.step()
            # Save model every 10 steps
            if (epoch_idx) % 10 == 0:
                # Transfer weights to MLP Net and save
                save_path = os.path.join(self.config.model_dir, f"nnmodel_e{epoch_idx}.pth")
                self.save_model(save_path)
        self.save_loss_plot(traj)

    def save_loss_plot(self, traj):
        loss_path = os.path.join(self.config.base_dir, "loss.json")
        with open(loss_path, "w") as f:
            test_maes = [x.item() for x in traj["test_maes"]]
            train_maes = [x.item() for x in traj["train_maes"]]
            new_json = {"test_maes": test_maes, "train_maes": train_maes}
            json.dump(new_json, f)
        # Save the training curve
        plot_path = os.path.join(self.config.base_dir, "training_curve.png")
        plt.plot(np.arange(len(traj["train_maes"])), traj["train_maes"], color="blue")
        plt.plot(np.arange(len(traj["test_maes"])), traj["test_maes"], color="red")
        plt.xlabel("Epochs")
        plt.ylabel("MAE (rad)")
        plt.title("MAE Curve")
        plt.savefig(plot_path)
        plt.close()

    def epoch(self):
        losses = []
        self.net.train()
        for image, gradient_map in tqdm(self.train_dataloader, position=0, leave=True, desc="Batch"):
            loss = self.training_step(gradient_map, image)
            losses.append(loss)
        self.net.eval()
        train_mean_loss = np.mean(np.concatenate(losses))
        val_mean_loss = evaluate(self.net, self.val_dataloader, self.device)

        if val_mean_loss < self.best_loss:
            self.best_loss = val_mean_loss
            self.save_model("best_model")
            logging.info(f"Best model saved")

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
    for image, gradient_map in tqdm(dataloader, desc="Eval Batch"):
        image = image.to(device)
        gradient_map = gradient_map.to(device)
        outputs = net(image)
        diffs = outputs - gradient_map
        losses.append(np.abs(diffs.cpu().detach().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


@draccus.wrap()
def main(config: TrainConfig):
    config_dir = os.path.join(config.base_dir, "config.yaml")
    draccus.dump(config, open(config_dir, "w"))
    trainer = Trainer(config)
    trainer.train_model()


if __name__ == "__main__":
    main()
