import os
import cv2
import json
import math
import numpy as np
from copy import deepcopy
import pickle
from typing import Any, NamedTuple
from PIL import Image
from termcolor import colored
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.builder import DATASET
from lib.utils.read_cameras import read_cameras_binary, read_images_binary


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


@DATASET.register_module()
class Our_Data:

    def __init__(self, cfg) -> None:
        self.name = type(self).__name__
        self.cfg = cfg

        self.obj_name = cfg.OBJ_NAME
        self.data_root = cfg.DATA_ROOT

        self.H = cfg.H
        self.W = cfg.W
        self.img_is_mask = cfg.get("IMG_IS_MASK", False)
        self.num_views = cfg.N_CAM
        self.all_frames = cfg.FRAME_LEN
        self.train_frames = cfg.TRAIN_FRAME
        self.eval_frames = cfg.EVAL_FRAME

        self.white_bkg = False
        self.bg = np.array([0, 0, 0])

        # Get the mask ids for the object from each view
        self.obj_idxes = []
        for i in range(self.num_views):
            with open(f"{self.data_root}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != "hand":
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            self.obj_idxes.append(obj_idx)

        logger.info(f"{self.name}: {colored(self.obj_name, 'yellow', attrs=['bold'])}")

    def get_colmap_info(self):
        # Get the camera parameters for the static stage images
        # Read the masked image and save it into PIL Image, save all into static_cams
        # R and T formulate the w2c matrix

        # Read the camera intrinsic
        with open(f"{self.data_root}/metadata.json", "r") as f:
            data = json.load(f)
        K_static = np.array(data["intrinsics"][0])

        # Read the camera extrinsics
        with open(f"{self.data_root}/calibrate.pkl", "rb") as file:
            c2ws = pickle.load(file)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        Rs = np.array([w2c[:3, :3] for w2c in w2cs])
        ts = np.array([w2c[:3, 3] for w2c in w2cs])

        # Read the three first-frame images
        img_paths = []
        mask_paths = []
        for i in range(self.num_views):
            img_path = f"{self.data_root}/color/{i}/0.png"
            img_paths.append(img_path)

            mask_path = f"{self.data_root}/mask/{i}/{self.obj_idxes[i]}/0.png"
            mask_paths.append(mask_path)

        static_cams = []
        self.static_nimgs = self.num_views

        for i in etqdm(range(self.num_views)):
            R = Rs[i]
            t = ts[i]
            image_path = img_paths[i]
            mask_path = mask_paths[i]

            image = Image.open(image_path)
            mask = Image.open(mask_path)

            im_data = np.array(image)
            height, width = im_data.shape[:2]
            mask = np.array(mask)[:, :, np.newaxis] / 255.0
            arr = (im_data / 255.0) * mask + self.bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            static_cams.append(
                CameraInfo(
                    uid=i,
                    R=np.transpose(R),
                    T=np.array(t),
                    FovY=focal2fov(K_static[1, 1], height),
                    FovX=focal2fov(K_static[0, 0], width),
                    image=image,
                    image_path=image_path,
                    image_name=f"{self.obj_name}_{i}_0.png",
                    width=width,
                    height=height,
                )
            )

        logger.info(
            f"{self.name}: {self.obj_name}, Got {colored(self.static_nimgs, 'yellow', attrs=['bold'])} images"
        )

        return static_cams

    def get_cam_info(self, img_is_mask=None):
        # Here image_is_mask is True
        # Read the intrinsic parameters from the static camera info (comap)
        if img_is_mask is None:
            img_is_mask = self.img_is_mask

        # Read the camera intrinsic
        with open(f"{self.data_root}/metadata.json", "r") as f:
            data = json.load(f)
        K = np.array(data["intrinsics"][0])

        FovY = focal2fov(K[1][1], self.H)
        FovX = focal2fov(K[0][0], self.W)

        # Read the camera extrinsics
        with open(f"{self.data_root}/calibrate.pkl", "rb") as file:
            c2ws = pickle.load(file)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        Rs = np.array([w2c[:3, :3] for w2c in w2cs])
        ts = np.array([w2c[:3, 3] for w2c in w2cs])

        logger.info("Reading all data ...")
        self.n_frames = self.train_frames
        logger.info(
            f"{self.name}: {self.obj_name}, Got {colored(self.n_frames, 'yellow', attrs=['bold'])} frames"
        )
        cam_infos_all = []

        for frame_id in etqdm(range(self.n_frames)):
            cam_infos = []
            for cam_idx in range(self.num_views):
                rot_mat = Rs[cam_idx]
                tvecs = ts[cam_idx]
                R = np.transpose(rot_mat)

                image_path = f"{self.data_root}/color/{cam_idx}/{frame_id}.png"
                mask_path = f"{self.data_root}/mask/{cam_idx}/{self.obj_idxes[cam_idx]}/{frame_id}.png"

                if img_is_mask:
                    mask = Image.open(mask_path)
                    image = Image.fromarray(
                        np.repeat(np.array(mask)[:, :, np.newaxis], 3, axis=-1), "RGB"
                    )
                else:
                    image = Image.open(image_path)
                    im_data = np.array(image)
                    mask = Image.open(mask_path)
                    mask = np.array(mask)[:, :, np.newaxis] / 255.0
                    arr = (im_data / 255.0) * mask + self.bg * (1 - mask)
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                cam_infos.append(
                    CameraInfo(
                        uid=cam_idx,
                        R=R,
                        T=tvecs,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        image_path=image_path,
                        image_name=f"{self.obj_name}_{cam_idx}_{frame_id}.png",
                        width=self.W,
                        height=self.H,
                    )
                )

            cam_infos_all.append(cam_infos)

        return cam_infos_all
