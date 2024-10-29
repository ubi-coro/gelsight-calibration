import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from calibration.utils import load_csv_as_dict, transfer_weights
from calibration.models import BGRXYDataset, BGRXYMLPNet_
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


def train_model():
    # Argument Parsers
    parser = argparse.ArgumentParser(description="Train the model from BGRXY to gxy.")
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="place where the calibration data is stored",
    )
    parser.add_argument(
        "-ne", "--n_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument("-lr", "--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="the device to train NN",
    )
    args = parser.parse_args()

    # Create the model directory
    calib_dir = args.calib_dir
    model_dir = os.path.join(calib_dir, "model")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Load the train and test split
    data_path = os.path.join(calib_dir, "train_test_split.json")
    with open(data_path, "r") as f:
        data = json.load(f)
        train_reldirs = data["train"]
        test_reldirs = data["test"]

    # Load the train and test data including the background data
    train_data = {"all_bgrxys": [], "all_gxyangles": []}
    for experiment_reldir in train_reldirs:
        data_path = os.path.join(calib_dir, experiment_reldir, "data.npz")
        if not os.path.isfile(data_path):
            raise ValueError("Data file %s does not exist" % data_path)
        data = np.load(data_path)
        train_data["all_bgrxys"].append(data["bgrxys"][data["mask"]])
        train_data["all_gxyangles"].append(data["gxyangles"][data["mask"]])
    test_data = {"all_bgrxys": [], "all_gxyangles": []}
    for experiment_reldir in test_reldirs:
        data_path = os.path.join(calib_dir, experiment_reldir, "data.npz")
        if not os.path.isfile(data_path):
            raise ValueError("Data file %s does not exist" % data_path)
        data = np.load(data_path)
        test_data["all_bgrxys"].append(data["bgrxys"][data["mask"]])
        test_data["all_gxyangles"].append(data["gxyangles"][data["mask"]])
    #Load background data
    bg_path = os.path.join(calib_dir, "background_data.npz")
    bg_data = np.load(bg_path)
    bgrxys = bg_data["bgrxys"][bg_data["mask"]]
    gxyangles = bg_data["gxyangles"][bg_data["mask"]]
    perm = np.random.permutation(len(bgrxys))
    n_train = np.sum([len(bgrxys) for bgrxys in train_data["all_bgrxys"]]) // 5
    n_test = np.sum([len(bgrxys) for bgrxys in test_data["all_bgrxys"]]) // 5
    if n_train + n_test > len(bgrxys):
        n_train = 4 * len(bgrxys) // 5
        n_test = len(bgrxys) // 5
    train_data["all_bgrxys"].append(bgrxys[perm[:n_train]])
    train_data["all_gxyangles"].append(gxyangles[perm[:n_train]])
    test_data["all_bgrxys"].append(bgrxys[perm[n_train : n_train + n_test]])
    test_data["all_gxyangles"].append(gxyangles[perm[n_train : n_train + n_test]])
    # Construct the train and test dataset
    train_bgrxys = np.concatenate(train_data["all_bgrxys"])
    train_gxyangles = np.concatenate(train_data["all_gxyangles"])
    test_bgrxys = np.concatenate(test_data["all_bgrxys"])
    test_gxyangles = np.concatenate(test_data["all_gxyangles"])

    # Create train and test Dataloader
    train_dataset = BGRXYDataset(train_bgrxys, train_gxyangles)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataset = BGRXYDataset(test_bgrxys, test_gxyangles)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Create the MLP Net for training
    device = args.device
    net = BGRXYMLPNet_().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create MLP Net in the CNN format for saving
    save_net = BGRXYMLPNet().to(device)

    # Initial evaluation
    train_mae = evaluate(net, train_dataloader, device)
    test_mae = evaluate(net, test_dataloader, device)
    naive_mae = np.mean(np.abs(test_gxyangles - np.mean(train_gxyangles, axis=0)))
    traj = {"train_maes": [train_mae], "test_maes": [test_mae], "naive_mae": naive_mae}
    print("Naive MAE (predict as mean): %.4f" % naive_mae)
    print("without train, Train MAE: %.4f, Test MAE: %.4f" % (train_mae, test_mae))

    # Train the model
    for epoch_idx in range(args.n_epochs):
        losses = []
        net.train()
        for bgrxys, gxyangles in train_dataloader:
            bgrxys = bgrxys.to(device)
            gxyangles = gxyangles.to(device)
            optimizer.zero_grad()
            outputs = net(bgrxys)
            loss = criterion(outputs, gxyangles)
            loss.backward()
            optimizer.step()
            diffs = outputs - gxyangles
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
            transfer_weights(net, save_net)
            save_path = os.path.join(model_dir, "nnmodel.pth")
            torch.save(save_net.state_dict(), save_path)

    # Save the training curve
    save_path = os.path.join(model_dir, "training_curve.png")
    plt.plot(np.arange(len(traj["train_maes"])), traj["train_maes"], color="blue")
    plt.plot(np.arange(len(traj["test_maes"])), traj["test_maes"], color="red")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (rad)")
    plt.title("MAE Curve")
    plt.savefig(save_path)
    plt.close()


def evaluate(net, dataloader, device):
    """
    Evaluate the network loss on the dataset.

    :param net: nn.Module; the network to evaluate.
    :param dataloader: DataLoader; the dataloader for the dataset.
    :param device: str; the device to evaluate the network.
    """
    losses = []
    for bgrxys, gxyangles in dataloader:
        bgrxys = bgrxys.to(device)
        gxyangles = gxyangles.to(device)
        outputs = net(bgrxys)
        diffs = outputs - gxyangles
        losses.append(np.abs(diffs.cpu().detach().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


if __name__ == "__main__":
    train_model()
