import os
import json
import numpy as np
import torch.optim as optim
import lightning as L
from torch import nn
from torch.utils.data import DataLoader
from models import GradientCNN, ColorGradientDataset



class GradientModel(L.LightningModule):
    def __init__(self, lr=0.002):
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
    def __init__(self, data_dir, batch_size=1):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        data_path = os.path.join(self.data_dir, "train_test_split.json")
        with open(data_path, "r") as f:
            data = json.load(f)
            train_dirs = [os.path.join(self.data_dir, rel_dir) for rel_dir in data["train"]]
            test_dirs = [os.path.join(self.data_dir, rel_dir) for rel_dir in data["test"]]

        train_data = self.load_data(train_dirs)
        test_data = self.load_data(test_dirs)

        self.train_dataset = ColorGradientDataset(train_data["images"], train_data["gradient_maps"])
        self.val_dataset = ColorGradientDataset(test_data["images"], test_data["gradient_maps"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def load_data(self, reldirs):
        data = {"images": [], "gradient_maps": []}
        for reldir in reldirs:
            path = os.path.join(reldir, "data.npz")
            if not os.path.isfile(path):
                raise ValueError(f"Data file {path} does not exist")
            loaded = np.load(path)
            data["gradient_maps"].append(loaded["gradient_map"])
            data["images"].append(loaded["image"])
        return data


def main():
    data_dir = "/media/local/data"
    batch_size = 10
    n_epochs = 50
    lr = 0.002

    data_module = GradientDataModule(data_dir, batch_size)
    model = GradientModel(lr)

    trainer = L.Trainer(max_epochs=n_epochs, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()