import json
import os
from dataclasses import dataclass

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from calibration.utils import transfer_weights
from models import ColorGradientDataset, GradientCNN
from gs_sdk.gs_reconstruct import BGRXYMLPNet

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
    # directory to the calibration data
    data_dir: str = "/media/local/data"
    # number of epochs. Default = 50
    n_epochs: int = 50
    # learning rate. Default = 0.002
    lr: float = 0.002
    # device to train the network. Must be "cuda" or "cpu". Default = "cuda"
    device: str = "cuda"

@draccus.wrap()
def train_model(config: TrainConfig):
    # Create the model directory
    calib_dir = config.data_dir
    model_dir = os.path.join(calib_dir, "model")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Load the train and test split
    data_path = os.path.join(calib_dir, "train_test_split.json")
    with open(data_path, "r") as f:
        data = json.load(f)
        train_dirs = [os.path.join(calib_dir, rel_dir) for rel_dir in data["train"]]
        test_dirs = [os.path.join(calib_dir, rel_dir) for rel_dir in data["test"]]

    train_data = load_data(train_dirs)
    test_data = load_data(test_dirs)

    # Construct the train and test dataset
    train_images = np.array(train_data["images"])
    train_gradient_maps = np.array(train_data["gradient_maps"])
    test_images = np.array(test_data["images"])
    test_gradient_maps = np.array(test_data["gradient_maps"])

    # Create train and test Dataloader
    train_dataset = ColorGradientDataset(train_images, train_gradient_maps)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = ColorGradientDataset(test_images, test_gradient_maps)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the CNN Net for training
    device = config.device
    net = GradientCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    # Initial evaluation
    train_mae = evaluate(net, train_dataloader, device)
    test_mae = evaluate(net, test_dataloader, device)
    naive_mae = np.mean(np.abs(test_gradient_maps - np.mean(train_gradient_maps, axis=0)))
    traj = {"train_maes": [train_mae], "test_maes": [test_mae], "naive_mae": naive_mae}
    print("Naive MAE (predict as mean): %.4f" % naive_mae)
    print("without train, Train MAE: %.4f, Test MAE: %.4f" % (train_mae, test_mae))

    # Train the model
    for epoch_idx in tqdm(range(config.n_epochs)):
        losses = []
        net.train()
        for image, gradient_map in train_dataloader:
            image = image.to(device)
            gradient_map = gradient_map.to(device)
            optimizer.zero_grad()
            outputs = net(image)
            loss = criterion(outputs, gradient_map)
            loss.backward()
            optimizer.step()
            diffs = outputs - gradient_map
            losses.append(np.abs(diffs.cpu().detach().numpy()))
        net.eval()
        traj["train_maes"].append(np.mean(np.concatenate(losses)))
        traj["test_maes"].append(evaluate(net, test_dataloader, device))
        print(
            "Epoch %i, Train MAE: %.4f, Test MAE: %.4f"
            % (epoch_idx, traj["train_maes"][-1], traj["test_maes"][-1])
        )
        scheduler.step()

        # Save model every 10 steps
        if (epoch_idx + 1) % 10 == 0:
            # Transfer weights to MLP Net and save
            save_path = os.path.join(model_dir, f"nnmodel_e{epoch_idx}.pth")
            torch.save(net.state_dict(), save_path)

    # Save the training curve
    save_path = os.path.join(model_dir, "training_curve.png")
    plt.plot(np.arange(len(traj["train_maes"])), traj["train_maes"], color="blue")
    plt.plot(np.arange(len(traj["test_maes"])), traj["test_maes"], color="red")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (rad)")
    plt.title("MAE Curve")
    plt.savefig(save_path)
    plt.close()

def load_data(reldirs):
    data = {"images": [], "gradient_maps": []}
    for reldir in reldirs:
        path = os.path.join(reldir, "data.npz")
        if not os.path.isfile(path):
            raise ValueError(f"Data file {path} does not exist")
        loaded = np.load(path)
        data["gradient_maps"].append(loaded["gradient_map"])
        data["images"].append(loaded["image"])
    return data

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
    train_model()
