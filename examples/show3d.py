import argparse
import time
from dataclasses import dataclass, field

import cv2
import os

import draccus
import numpy as np
import yaml

from calibrationUNET.net_factory import NetName
from gs_sdk import gs_reconstruct
from gs_sdk.gs_device import Camera
from gs_sdk.gs_reconstruct import calc_depth_map, Reconstructor
from gs_sdk.gs_reconstructCNN import ReconstructorCNN
from gs_sdk.viz_utils import gradient_img, depth_img
from image3d import depth_map_to_point_cloud

config_dir = os.path.join(os.path.dirname(__file__), "./configs")
model_dir = os.path.join(os.path.dirname(__file__), "./models")

@dataclass
class Args:
    training_dir: str = field()
    background_path: str = field()

    device: str = field(default="cpu")
    device_config_path: str = os.path.join(config_dir, "gsmini_highres.yaml")
    model_name: str = "best_model.pth"
    save_dir: str = "save"

    @property
    def model_path(self):
            return os.path.join(self.training_dir, "model", self.model_name)

    @property
    def net_name(self):
        config_path = os.path.join(self.training_dir, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            net_name = config["net_name"]
        return net_name

@draccus.wrap()
def get_reconstructor(args: Args):
    if args.net_name == NetName.PixelNet:
        return Reconstructor(args.model_path, device=args.device)
    elif args.net_name == NetName.GradientCNN:
        return ReconstructorCNN(args.model_path, device=args.device)

@draccus.wrap()
def main(args: Args):

    with open(args.device_config_path, "r") as f:
        config = yaml.safe_load(f)
        # Get the camera resolution
        ppmm = config["ppmm"]
        imgw = config["imgw"]
        imgh = config["imgh"]

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = Camera("GelSight Mini", imgh, imgw)
    dev.connect()


    recon = get_reconstructor()
    bg_image = cv2.imread(args.background_path)
    recon.load_bg(bg_image)

    print('press q on image to exit')

    try:
        while True:

            # get the roi image
            f1 = dev.get_image()
            cv2.imshow("image", f1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('q')
                break
            elif key == ord('s'):
                print("save")
                # compute the depth map
                G, H_poisson, C = recon.get_surface_info(f1, ppmm)
                H2 = calc_depth_map(G[:,:,0], G[:,:,1])
                time_stamp = time.time()
                save_path_img = os.path.join(args.save_dir, f"{time_stamp}.png")
                save_path_depth_poisson = os.path.join(args.save_dir, f"{time_stamp}_depth_poisson.png")
                save_path_depth = os.path.join(args.save_dir, f"{time_stamp}_depth.png")
                save_path_G = os.path.join(args.save_dir, f"{time_stamp}_gradients.png")
                save_path_C = os.path.join(args.save_dir, f"{time_stamp}_contact.png")

                grad_img = gradient_img(G[:,:, 0], G[:, :, 1])
                cv2.imwrite(save_path_img, f1)
                cv2.imwrite(save_path_depth_poisson, depth_img(H_poisson))
                cv2.imwrite(save_path_depth, depth_img(H2))
                cv2.imwrite(save_path_G, grad_img)
                cv2.imwrite(save_path_C, C.astype(np.uint8)*255)
                depth_map_to_point_cloud(H_poisson, vis=True)
                depth_map_to_point_cloud(H2, vis=True)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
