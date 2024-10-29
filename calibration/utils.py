import csv
import numpy as np


def load_csv_as_dict(csv_path):
    """
    Load the csv file entries as dictionaries.

    :params csv_path: str; the path of the csv file.
    :returns: dict; the dictionary of the csv file.
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        data = list(reader)
        keys = reader.fieldnames
        data_dict = {}
        for key in keys:
            data_dict[key] = []
        for line in data:
            for key in keys:
                data_dict[key].append(line[key])
    return data_dict


def transfer_weights(mlp_model, fcn_model):
    """
    transfer weights between BGRXYMLPNet_ to BGRXYMLPNet.

    :param mlp_model: BGRXYMLPNet_; the model to transfer from.
    :param fcn_model: BGRXYMLPNet; the model to transfer to.
    """
    # Copy weights from fc1 to conv1
    fcn_model.conv1.weight.data = mlp_model.fc1.weight.data.view(
        fcn_model.conv1.weight.size()
    )
    fcn_model.conv1.bias.data = mlp_model.fc1.bias.data
    fcn_model.bn1.weight.data = mlp_model.bn1.weight.data
    fcn_model.bn1.bias.data = mlp_model.bn1.bias.data
    fcn_model.bn1.running_mean = mlp_model.bn1.running_mean
    fcn_model.bn1.running_var = mlp_model.bn1.running_var
    # Copy weights from fc2 to conv2
    fcn_model.conv2.weight.data = mlp_model.fc2.weight.data.view(
        fcn_model.conv2.weight.size()
    )
    fcn_model.conv2.bias.data = mlp_model.fc2.bias.data
    fcn_model.bn2.weight.data = mlp_model.bn2.weight.data
    fcn_model.bn2.bias.data = mlp_model.bn2.bias.data
    fcn_model.bn2.running_mean = mlp_model.bn2.running_mean
    fcn_model.bn2.running_var = mlp_model.bn2.running_var
    # Copy weights from fc3 to conv3
    fcn_model.conv3.weight.data = mlp_model.fc3.weight.data.view(
        fcn_model.conv3.weight.size()
    )
    fcn_model.conv3.bias.data = mlp_model.fc3.bias.data
    fcn_model.bn3.weight.data = mlp_model.bn3.weight.data
    fcn_model.bn3.bias.data = mlp_model.bn3.bias.data
    fcn_model.bn3.running_mean = mlp_model.bn3.running_mean
    fcn_model.bn3.running_var = mlp_model.bn3.running_var
    # Copy weights from fc4 to conv4
    fcn_model.conv4.weight.data = mlp_model.fc4.weight.data.view(
        fcn_model.conv4.weight.size()
    )
    fcn_model.conv4.bias.data = mlp_model.fc4.bias.data
    return fcn_model
