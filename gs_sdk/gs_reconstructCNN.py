import math
import os

import cv2
import numpy as np
import open3d
from scipy import fftpack
import torch

from gs_sdk.gs_reconstruct import Reconstructor
from calibrationUNET.models import GradientCNN

class ReconstructorCNN:
    """
    The GelSight reconstruction class.

    This class handles 3D reconstruction from calibrated GelSight images.
    """

    def __init__(self, model_path, contact_mode="standard", device="cpu"):
        """
        Initialize the reconstruction model.
        Contact mode "flat" means the object in contact is flat, so a different threshold
        is used to determine contact mask.

        :param model_path: str; the path of the calibrated neural network model.
        :param contact_mode: str {"standard", "flat"}; the mode to get the contact mask.
        :param device: str {"cuda", "cpu"}; the device to run the model.
        """
        self.model_path = model_path
        self.contact_mode = contact_mode
        self.device = device
        self.bg_image = None
        # Load the gxy model
        if not os.path.isfile(model_path):
            raise ValueError("Error opening %s, file does not exist" % model_path)
        self.gxy_net = GradientCNN()
        self.gxy_net.load_state_dict(torch.load(model_path), self.device)
        self.gxy_net.eval()

    def load_bg(self, bg_image):
        """
        Load the background image.

        :param bg_image: np.array (H, W, 3); the background image.
        """
        self.bg_image = bg_image

        # Calculate the gradients of the background
        bgrxys = image2bgrxys(bg_image)
        bgrxys = bgrxys.transpose(2, 0, 1)
        features = torch.from_numpy(bgrxys[np.newaxis, :, :, :]).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles[0].cpu().detach().numpy()
            self.bg_G = np.tan(gxyangles.transpose(1, 2, 0))

    def get_surface_info(self, image, ppmm):
        """
        Get the surface information including gradients (G), height map (H), and contact mask (C).

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :return G: np.array (H, W, 2); the gradients.
                H: np.array (H, W); the height map.
                C: np.array (H, W); the contact mask.
        """
        # Calculate the gradients
        bgrxys = image2bgrxys(image)
        bgrxys = bgrxys.transpose(2, 0, 1)
        features = torch.from_numpy(bgrxys[np.newaxis, :, :, :]).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles[0].cpu().detach().numpy()
            G = np.tan(gxyangles.transpose(1, 2, 0))
            if self.bg_image is not None:
                G = G - self.bg_G
            else:
                raise ValueError("Background image is not loaded.")

        # Calculate the height map
        H = poisson_dct_neumaan(G[:, :, 0], G[:, :, 1]).astype(np.float32)

        # Calculate the contact mask
        if self.contact_mode == "standard":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 15
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            )
            color_mask = cv2.erode(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )

            # Filter by height
            cutoff = np.percentile(H, 85) - 0.2 / ppmm
            height_mask = H < cutoff
            C = np.logical_and(color_mask, height_mask)
        elif self.contact_mode == "flat":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 10
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            C = cv2.erode(
                color_mask.astype(np.uint8), np.ones((25, 25), np.uint8)
            ).astype(np.bool_)

        return G, H, C


def image2bgrxys(image):
    """
    Convert a bgr image to bgrxy feature.

    :param image: np.array (H, W, 3); the bgr image.
    :return: np.array (H, W, 5); the bgrxy feature.
    """
    xys = np.dstack(
        np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy")
    )
    xys = xys.astype(np.float32) / np.array([image.shape[1], image.shape[0]])
    bgrs = image.copy() / 255
    bgrxys = np.concatenate([bgrs, xys], axis=2)
    return bgrxys


def poisson_dct_neumaan(gx, gy):
    """
    2D integration of depth from gx, gy using Poisson solver.

    :param gx: np.array (H, W); the x gradient.
    :param gy: np.array (H, W); the y gradient.
    :return: np.array (H, W); the depth map.
    """
    # Compute Laplacian
    gxx = 1 * (
        gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])]
        - gx[:, ([0] + list(range(gx.shape[1] - 1)))]
    )
    gyy = 1 * (
        gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :]
        - gy[([0] + list(range(gx.shape[0] - 1))), :]
    )
    f = gxx + gyy

    # Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    # Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    # Modification near the corners (Eq. 54 in [1])
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    # Cosine transform of f
    tt = fftpack.dct(f, norm="ortho")
    fcos = fftpack.dct(tt.T, norm="ortho").T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    )

    # Inverse Discrete cosine Transform
    f = -fcos / denom
    tt = fftpack.idct(f, norm="ortho")
    img_tt = fftpack.idct(tt.T, norm="ortho").T
    img_tt = img_tt.mean() + img_tt

    return img_tt

def calc_depth_map(gx, gy):
    """
    :param gx: np.array (H, W); the x gradient.
    :param gy: np.array (H, W); the y gradient.
    :return: np.array (H, W); the integrated depth map.
    """
    H, W = gx.shape

    # Integrate along the x-axis
    depth_map_x = np.zeros((H, W))
    for i in range(1, W):
        depth_map_x[:, i] = depth_map_x[:, i - 1] + gx[:, i - 1]

    # Integrate along the y-axis
    depth_map_y = np.zeros((H, W))
    for j in range(1, H):
        depth_map_y[j, :] = depth_map_y[j - 1, :] + gy[j - 1, :]

    # Combine both depth maps (e.g., by averaging)
    depth_map = 0.5 * (depth_map_x + depth_map_y)

    return depth_map

class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m
        self.init_open3D()
        self.cnt = 212
        self.save_path = save_path
        pass

    def init_open3D(self):
        x = np.arange(self.n)# * mmpp
        y = np.arange(self.m)# * mmpp
        self.X, self.Y = np.meshgrid(x,y)
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X) #/ self.m
        self.points[:, 1] = np.ndarray.flatten(self.Y) #/ self.n

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z):
        self.depth2points(Z)
        dx, dy = np.gradient(Z)
        dx, dy = dx * 0.5, dy * 0.5

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3): colors[:,_]  = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        #### SAVE POINT CLOUD TO A FILE
        if self.save_path != '':
            open3d.io.write_point_cloud(self.save_path + "/pc_{}.pcd".format(self.cnt), self.pcd)

        self.cnt += 1

    def save_pointcloud(self):
        open3d.io.write_point_cloud(self.save_path + "pc_{}.pcd".format(self.cnt), self.pcd)
