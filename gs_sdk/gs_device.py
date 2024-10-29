import os
import re
import subprocess

import cv2
import ffmpeg
import numpy as np


class Camera:
    """
    The GelSight Camera Class.

    This class handles camera initialization, image acquisition, and camera release.
    Some sensors (GelSight Mini) might experience frame dropping issues, use FastCamera class instead.
    """

    def __init__(self, dev_type, imgh, imgw):
        """
        Initialize the camera.

        :param dev_type: str; The type of the camera.
        :param imgh: int; The height of the image.
        :param imgw: int; The width of the image.
        """
        self.dev_type = dev_type
        self.dev_id = get_camera_id(self.dev_type)
        self.imgh = imgh
        self.imgw = imgw
        self.cam = None
        self.data = None

    def connect(self):
        """
        Connect to the camera using cv2 streamer.
        """
        self.cam = cv2.VideoCapture(self.dev_id)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cam is None or not self.cam.isOpened():
            print("Warning: unable to open video source %d" % (self.dev_id))
        else:
            print("Connect to %s at video source %d" % (self.dev_type, self.dev_id))

    def get_image(self, flush=False):
        """
        Get the image from the camera.

        :param flush: bool; Whether to flush the first few frames.
        :return: np.ndarray; The image from the camera.
        """
        if flush:
            # flush out fist few frames to remove black frames
            for i in range(10):
                ret, f0 = self.cam.read()
        ret, f0 = self.cam.read()
        if ret:
            f0 = resize_crop(f0, self.imgw, self.imgh)
            self.data = f0
        else:
            print("ERROR! reading image from video source %d" % (self.dev_id))
        return self.data

    def release(self):
        """
        Release the camera resource.
        """
        if self.cam is not None:
            self.cam.release()
            print("Video source %d released." % (self.dev_id))
        else:
            print("No camera to release.")


class FastCamera:
    """
    The GelSight Camera Class with low latency.

    This class handles camera initialization, image acquisition, and camera release with low latency.
    """

    def __init__(self, dev_type, imgh, imgw, raw_imgh, raw_imgw, framerate):
        """
        Initialize the low latency camera. Raw camera parameters are required to stream with low latency.

        :param dev_type: str; The type of the camera.
        :param imgh: int; The desired height of the image.
        :param imgw: int; The desired width of the image.
        :param raw_imgh: int; The raw height of the image.
        :param raw_imgw: int; The raw width of the image.
        :param framerate: int; The frame rate of the camera.
        """
        # Raw image size
        self.raw_imgh = raw_imgh
        self.raw_imgw = raw_imgw
        self.raw_size = self.raw_imgh * self.raw_imgw * 3
        self.framerate = framerate
        # desired image size
        self.imgh = imgh
        self.imgw = imgw
        # Get camera ID
        self.dev_type = dev_type
        self.dev_id = get_camera_id(self.dev_type)
        self.device = "/dev/video" + str(self.dev_id)

    def connect(self):
        """
        Connect to the camera using FFMpeg streamer.
        """
        # Command to capture video using ffmpeg and high resolution
        self.ffmpeg_command = (
            ffmpeg.input(
                self.device,
                format="v4l2",
                framerate=self.framerate,
                video_size="%dx%d" % (self.raw_imgw, self.raw_imgh),
            )
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .global_args("-fflags", "nobuffer")
            .global_args("-flags", "low_delay")
            .global_args("-fflags", "+genpts")
            .global_args("-rtbufsize", "0")
            .compile()
        )
        self.process = subprocess.Popen(
            self.ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Warm-up phase: discard the first few frames
        print("Warming up the camera...")
        warm_up_frames = 100
        for _ in range(warm_up_frames):
            self.process.stdout.read(self.raw_size)
        print("Camera ready for use!")

    def get_image(self):
        """
        Get the image from the camera from raw data stream.

        :return: np.ndarray; The image from the camera.
        """
        raw_frame = self.process.stdout.read(self.raw_size)
        frame = np.frombuffer(raw_frame, np.uint8).reshape(
            (self.raw_imgh, self.raw_imgw, 3)
        )
        frame = resize_crop(frame, self.imgw, self.imgh)
        return frame

    def release(self):
        """
        Release the camera resource.
        """
        self.process.stdout.close()
        self.process.wait()


def get_camera_id(camera_name):
    """
    Find the camera ID that has the corresponding camera name.

    :param camera_name: str; The name of the camera.
    :return: int; The camera ID.
    """
    cam_num = None
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
        if camera_name in name:
            cam_num = int(re.search("\d+$", file).group(0))
            found = "FOUND!"
        else:
            found = "      "
        print("{} {} -> {}".format(found, file, name))

    return cam_num


def resize_crop(img, imgw, imgh):
    """
    Resize and crop the image to the desired size.

    :param img: np.ndarray; The image to resize and crop.
    :param imgw: int; The width of the desired image.
    :param imgh: int; The height of the desired image.
    :return: np.ndarray; The resized and cropped image.
    """
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    # Extra cropping to maintain aspect ratio
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    # keep the ratio the same as the original image size
    img = img[
        border_size_x + extra_border_h : img.shape[0] - border_size_x,
        border_size_y + extra_border_w : img.shape[1] - border_size_y,
    ]
    # final resize for the desired image size
    img = cv2.resize(img, (imgw, imgh))
    return img
