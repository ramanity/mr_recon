import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils import data
from scipy import ndimage as nd
from tqdm import tqdm

def read_img(in_path):
    """
    Read MR images from a specified directory.

    This function reads all images in the specified directory
    and converts them into numpy arrays.

    Parameters
    ----------
    in_path : str
        Path to the directory containing the images.

    Returns
    -------
    list
        List of image volumes as numpy arrays.
    """
    img_list = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_list.append(img_vol)
    return img_list

def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.

    This function generates coordinates at grid centers
    for the given shape.

    Parameters
    ----------
    shape : tuple
        Shape of the grid.
    ranges : tuple, optional
        Ranges for each dimension. Default is None.
    flatten : bool, optional
        Flatten the coordinates or not. Default is True.

    Returns
    -------
    torch.Tensor
        Coordinates of the grid.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret.view(-1, ret.shape[-1]) if flatten else ret

class ImgTrain(data.Dataset):
    """
    Create a custom dataset for training.

    This class represents a dataset of high-resolution image patches
    for training a neural network.

    Parameters
    ----------
    in_path_hr : str
        Path to the high-resolution image patches.
    sample_size : int
        Number of samples to draw.
    is_train : bool
        Indicator if the dataset is for training or not.
    """
    def __init__(self, in_path_hr, sample_size, is_train):
        self.is_train = is_train
        self.sample_size = sample_size
        self.patch_hr = read_img(in_path=in_path_hr)

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        """
        Get a single data item.

        This method fetches a high-resolution image patch,
        down-samples it to create a low-resolution patch,
        and generates the corresponding coordinates.

        Parameters
        ----------
        item : int
            Index of the item to fetch.

        Returns
        -------
        tuple
            Low-resolution patch, coordinates, and high-resolution patch.
        """
        patch_hr = self.patch_hr[item]
        
        # Randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        
        # Compute the size of HR patch according to the scale
        hr_h, hr_w, hr_d = (np.array([10, 10, 10]) * s).astype(int)
        
        # Generate HR patch by cropping
        patch_hr = patch_hr[:hr_h, :hr_w, :hr_d]
        
        # Simulate LR patch by down-sampling HR patch
        patch_lr = nd.zoom(patch_hr, 1 / s, order=3)
        
        # Generate coordinate set
        xyz_hr = make_coord(patch_hr.shape, flatten=True)
        
        # Randomly sample voxel coordinates
        if self.is_train:
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
            
        return patch_lr, xyz_hr, patch_hr

def loader_train(in_path_hr, batch_size, sample_size, is_train):
    """
    Create a DataLoader for the training dataset.

    Parameters
    ----------
    in_path_hr : str
        Path to high-resolution image patches.
    batch_size : int
        Batch size for training.
    sample_size : int
        Number of samples to draw.
    is_train : bool
        Indicator if the loader is for training or not.

    Returns
    -------
    DataLoader
        DataLoader for the training dataset.
    """
    return data.DataLoader(
        dataset=ImgTrain(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )

class ImgTest(data.Dataset):
    """
    Custom dataset for testing.

    This class represents a dataset of low-resolution images
    for testing a neural network.

    Parameters
    ----------
    in_path_lr : str
        Path to the low-resolution image.
    scale : float
        Scale factor for up-sampling.
    """
    def __init__(self, in_path_lr, scale):
        self.img_lr = []
        self.xyz_hr = []
        lr_vol = sitk.GetArrayFromImage(sitk.ReadImage(in_path_lr))
        self.img_lr.append(lr_vol)
        for img_lr in self.img_lr:
            temp_size = (np.array(img_lr.shape) * scale).astype(int)
            self.xyz_hr.append(make_coord(temp_size, flatten=True))

    def __len__(self):
        return len(self.img_lr)

    def __getitem__(self, item):
        """
        Get a single data item.

        This method fetches a low-resolution image and its corresponding
        high-resolution coordinates.

        Parameters
        ----------
        item : int
            Index of the item to fetch.

        Returns
        -------
        tuple
            Low-resolution image and high-resolution coordinates.
        """
        return self.img_lr[item], self.xyz_hr[item]

def loader_test(in_path_lr, scale):
    """
    Create a data loader for the testing dataset.

    Parameters
    ----------
    in_path_lr : str
        Path to the low-resolution image.
    scale : float
        Scale factor for up-sampling.

    Returns
    -------
    DataLoader
        DataLoader for the testing dataset.
    """
    return data.DataLoader(
        dataset=ImgTest(in_path_lr=in_path_lr, scale=scale),
        batch_size=1,
        shuffle=False
    )
