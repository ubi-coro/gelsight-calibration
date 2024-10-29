import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BGRXYDataset(Dataset):
    """The BGRXY Datast."""

    def __init__(self, bgrxys, gxyangles):
        self.bgrxys = bgrxys
        self.gxyangles = gxyangles

    def __len__(self):
        return len(self.bgrxys)

    def __getitem__(self, index):
        bgrxy = torch.tensor(self.bgrxys[index], dtype=torch.float32)
        gxyangles = torch.tensor(self.gxyangles[index], dtype=torch.float32)
        return bgrxy, gxyangles


class BGRXYMLPNet_(nn.Module):
    """
    The architecture using MLP, this is never used in test time.
    We train with this architecture and then transfer weights to the 1-by-1 convolution architecture.
    """

    def __init__(self):
        super(BGRXYMLPNet_, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
