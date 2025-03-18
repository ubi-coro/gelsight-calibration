import argparse
import time
from argparse import Namespace

import cv2
import os

import numpy as np
import yaml

from gs_sdk import gs_device
from gs_sdk import gs_reconstruct
from gs_sdk.gs_reconstruct import calc_depth_map
from gs_sdk.viz_utils import gradient_img, depth_img

config_dir = os.path.join(os.path.dirname(__file__), "./configs")
model_dir = os.path.join(os.path.dirname(__file__), "./models")

from image3d import depth_map_to_point_cloud

def main():
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False

    # Path to 3d model
    args: Namespace = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        # Get the camera resolution
        ppmm = config["mmpp"]
        imgw = config["imgw"]
        imgh = config["imgh"]

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gs_device.Camera("GelSight Mini", imgh, imgw)
    dev.connect()

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    recon = gs_reconstruct.Reconstructor(args.model_path, device=gpuorcpu)
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



def parse_args() -> Namespace:
    # Argument Parser
    parser = argparse.ArgumentParser(description="Show 3D-Reconstruction")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to load and run the neural network model.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the sensor information",
        default=os.path.join(config_dir, "gsmini_highres.yaml"),
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="path of the model",
        default=os.path.join(model_dir, "gsmini.pth"),
    )
    parser.add_argument(
        "-bg",
        "--background_path",
        type=str,
        help="path of the background image",
        default=os.path.join(model_dir, "background.png"),
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        help="path of the save directory",
        default=os.path.join(".", "save"),
    )
    args: Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
