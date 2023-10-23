import logging
import os.path
#from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
import h5py
# from copy import deepcopy
# from random import randrange
import json

# import numpy
# import torch

import albumentations as A
import numpy as np
import scipy
import volumentations as V
# import yaml

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OpmotionDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="opmotion",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[str] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.80171832, 0.7446163, 0.67687815),
            (0.28270636, 0.3032189, 0.33863187),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        max_cut_region=0,
        point_per_cut=100,
        cache_data=True,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        filter_out_classes=[],
        label_offset=0,
        is_elastic_distortion=False,
    ):

        self.is_elastic_distortion = is_elastic_distortion
        self.dataset_name = dataset_name
        # self.reps_per_epoch = reps_per_epoch
        self.label_offset = label_offset
        self.mode = mode
        # self.data_dir = data_dir
        #
        # if type(data_dir) == str:
        #     self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut

        # loading database files
        self._data = []

        f = h5py.File("data/processed/opmotion/opmotion.h5", "r")
        semantic_ids = f["semantic_ids"][()]
        instance_ids = f["instance_ids"][()]
        points = f["points"][()]
        rgb = f["colors"][()]
        normal = f["normals"][()]
        model_ids = f["model_ids"][()]
        with open("data/processed/opmotion/data.json" ,"r") as f:
            tmp_mode = mode
            if mode == "validation":
                tmp_mode = "val"
            split_ids = json.load(f)[tmp_mode]
        for i, model_id in enumerate(model_ids):
            # database_path = Path(database_path)
            # self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))
            if str(int(model_ids[i])) not in split_ids:
                continue
            self._data.append({
                "data": np.hstack((points[i], rgb[i], normal[i], semantic_ids[i][..., np.newaxis], instance_ids[i][..., np.newaxis])),
                "model_id": str(int(model_ids[i]))
            })

            os.makedirs(f"data/processed/opmotion/instance_gt/{mode}", exist_ok=True)
            gt_path = f"data/processed/opmotion/instance_gt/{mode}/{str(int(model_ids[i]))}.txt"
            gt_data = semantic_ids[i] * 1000 + instance_ids[i] + 1
            if not os.path.exists(gt_path):

                np.savetxt(gt_path, gt_data.astype(np.int32), fmt="%d")

        # if working only on classes for validation - discard others
        self._labels = {
            1: {"name": "drawer", "validation": True},
            2: {"name": "door", "validation": True},
            3: {"name": "lid", "validation": True},
            4: {"name": "base", "validation": True}
        }

        color_mean, color_std = color_mean_std[0], color_mean_std[1]

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (volume_augmentations_path != "none"):
            self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml")
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (image_augmentations_path != "none"):
            self.image_augmentations = A.load(Path(image_augmentations_path), data_format="yaml")
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):

        points = self.data[idx]["data"]

        coordinates, color, normals, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9:11]
        )

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train

        if "train" in self.mode:

            coordinates -= coordinates.mean(0)
            # coordinates += (np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2)

            # for i in (0, 1):
            #     if random() < 0.5:
            #         coord_max = np.max(points[:, i])
            #         coordinates[:, i] = coord_max - coordinates[:, i]

            # if random() < 0.95:
            #     if self.is_elastic_distortion:
            #         for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
            #             coordinates = elastic_distortion(coordinates, granularity, magnitude)

            aug = self.volume_augmentations(points=coordinates, normals=normals, features=color, labels=labels)

            coordinates, color, normals, labels = (aug["points"], aug["features"], aug["normals"], aug["labels"])
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            # if self.point_per_cut != 0:
            #     number_of_cuts = int(len(coordinates) / self.point_per_cut)
            #     for _ in range(number_of_cuts):
            #         size_of_cut = np.random.uniform(0.05, self.max_cut_region)
            #         # not wall, floor or empty
            #         point = choice(coordinates)
            #         x_min = point[0] - size_of_cut
            #         x_max = x_min + size_of_cut
            #         y_min = point[1] - size_of_cut
            #         y_max = y_min + size_of_cut
            #         z_min = point[2] - size_of_cut
            #         z_max = z_min + size_of_cut
            #         indexes = crop(coordinates, x_min, y_min, z_min, x_max, y_max, z_max)
            #         coordinates, normals, color, labels = (
            #             coordinates[~indexes],
            #             normals[~indexes],
            #             color[~indexes],
            #             labels[~indexes],
            #         )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        # if labels.size > 0:
        #     labels[:, 0] = self._remap_from_zero(labels[:, 0])
            # if not self.add_instance:
            #     # taking only first column, which is segmentation label, not instance
            #     labels = labels[:, 0].flatten()[..., None]

        segments = np.zeros(shape=(labels.shape[0]), dtype=np.float32)
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        # if self.add_normals:
        #     features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        return (coordinates, features, labels, self.data[idx]["model_id"], raw_color, raw_normals, raw_coordinates, idx)

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels


    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud
