import psutil
import os
import torch
import matplotlib.pyplot as plt
import mat73
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join 
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Sequence, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from threading import Lock


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss/(1e3)**3
    print('Memory Usage in Gb: {:.2f}'.format(mem))  # in GB
    return mem


def split_tensor(data: torch.Tensor, 
                 split_ratios: Union[None, Tuple[float, float, float]] = (0.8, 0.1, 0.1),
                 split_nums: Union[None, Tuple[int, int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a tensor into training, validation, and test sets.

    Parameters:
    - data (torch.Tensor): The input tensor to be split.
    - split_ratios (Tuple[float, float, float], optional): The ratios to split the data.
    - split_nums (Tuple[int, int, int], optional): The number of samples in each split.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Training, validation, and test sets.
    """
    
    num_samples = data.shape[0]

    if split_nums is not None:
        assert sum(split_nums) == num_samples, "Sum of split_nums must equal the number of samples."
        num_train, num_valid, num_test = split_nums
    else:
        assert sum(split_ratios) == 1.0, "Sum of split_ratios must be 1.0."
        num_train = int(split_ratios[0] * num_samples)
        num_valid = int(split_ratios[1] * num_samples)
        num_test = num_samples - num_train - num_valid

    train_data = data[:num_train]
    valid_data = data[num_train:num_train + num_valid]
    test_data = data[num_train + num_valid:]

    return train_data, valid_data, test_data


def spatial_cut_and_stack(x_data, spatial_dims, nx, ny, nz):
    # self.x_data: (N, X, Y, Z)
    x_fold = x_data.shape[1]//nx
    y_fold = x_data.shape[2]//ny
    z_fold = x_data.shape[3]//nz

    x_data_cropped = np.vstack(
        [x_data[:, i*nx:(i+1)*nx, j*ny:(j+1)*ny, k*nz:(k+1)*nz] 
         for i in range(x_fold) for j in range(y_fold) for k in range(z_fold)
         ]
        ) # (N*(X//nx)*(Y//ny)*(Z//nz), nx, ny, nz)

    if spatial_dims == 3: 
        x_data = x_data_cropped[:, None] # (N, 1, X, Y, Z)
    elif spatial_dims == 2:
        x_data = np.vstack(
            x_data_cropped.transpose((0, -1, 1, 2))
            )[:, None, ..., None] # (NZ, 1, X, Y, 1)
    x_data = torch.tensor(
        x_data[..., :nx, :ny, :], dtype=torch.float32)
    return x_data


def collate_with_augmentation(batch, augmentor):
    # Set augmentation parameters for the entire batch
    augmentor.set_augmentation_params()

    # Augment each sample in the batch using the same parameters
    augmented_batch = [augmentor.augment(sample) for sample in batch]
    return torch.stack(augmented_batch)


class RelativeError(object):
    def __init__(self, p=2, d=None):
        """
        Initializes the RelativeError instance.
        
        Parameters:
        p (int): The norm type to use (e.g., L2 norm where p=2). Must be > 0.
        d (int or tuple of ints, optional): The dimension(s) over which to compute the norm. Default is None (compute over the entire tensor).
        """
        assert p > 0, "Lp-norm type `p` must be greater than 0"  # Ensure p is positive
        self.p = p  # Lp-norm type
        self.d = d  # Dimension along which to compute the norm

    def __call__(self, y_pred, y, epsilon=1e-8):
        """
        Computes the relative error between predicted and true values.
        
        Parameters:
        y_pred (torch.Tensor): Predicted tensor.
        y (torch.Tensor): True tensor.
        epsilon (float): Small value added to prevent division by zero. Default is 1e-8.
        
        Returns:
        torch.Tensor: Mean relative error across the specified dimension.
        """

        diff_norms = torch.norm(y_pred - y, p=self.p, dim=self.d) # Compute the Lp-norm of the difference
        y_norms = torch.norm(y, p=self.p, dim=self.d) + epsilon # Compute the Lp-norm of the true values, with epsilon to avoid division by zero
        return torch.mean(diff_norms / y_norms) # Compute the relative error and return the mean value


class DataAugmentor:
    def __init__(self, 
                 crop_sizes=None, 
                 flip_directions=None, 
                 rotation_angles=None, 
                 augmentation_prob=0.5, 
                 fixed_params=None):
        
        self.crop_sizes = crop_sizes
        self.flip_directions = flip_directions
        self.rotation_angles = rotation_angles
        self.augmentation_prob = augmentation_prob
        self.fixed_params = fixed_params  # Add fixed_params to hold fixed settings

    def set_augmentation_params(self):
        if self.fixed_params:
            # Use fixed parameters if provided
            self.should_augment = True
            self.selected_crop_size = self.fixed_params.get("crop_size", None)
            self.selected_flip_direction = self.fixed_params.get("flip_direction", None)
            self.selected_rotation_angle = self.fixed_params.get("rotation_angle", None)
        else:
            # Randomly decide parameters for the batch
            self.should_augment = random.random() < self.augmentation_prob
            if self.should_augment:
                self.selected_crop_size = random.choice(self.crop_sizes) if self.crop_sizes else None
                self.selected_flip_direction = random.choice(self.flip_directions) if self.flip_directions else None
                self.selected_rotation_angle = random.choice(self.rotation_angles) if self.rotation_angles else None

    def augment(self, sample):
        if not self.should_augment:
            return sample

        if self.selected_crop_size:
            sample = self.random_crop(sample, self.selected_crop_size)
        if self.selected_flip_direction:
            sample = self.random_flip(sample, self.selected_flip_direction)
        if self.selected_rotation_angle:
            sample = self.random_rotate(sample, self.selected_rotation_angle)
        return sample

    def random_crop(self, sample, crop_size):
        _, nx, ny, nz = sample.shape
        cropped_nx, cropped_ny, cropped_nz = crop_size

        x_start = random.randint(0, nx - cropped_nx) if cropped_nx < nx else 0
        y_start = random.randint(0, ny - cropped_ny) if cropped_ny < ny else 0
        z_start = random.randint(0, nz - cropped_nz) if cropped_nz < nz else 0

        return sample[:, x_start:x_start+cropped_nx, y_start:y_start+cropped_ny, z_start:z_start+cropped_nz]

    def random_flip(self, sample, direction):
        if direction == 'x':
            sample = torch.flip(sample, dims=[1])  # Flip along x-axis
        elif direction == 'y':
            sample = torch.flip(sample, dims=[2])  # Flip along y-axis
        elif direction == 'xy':
            sample = torch.flip(sample, dims=[1, 2])  # Flip along both axes
        return sample

    def random_rotate(self, sample, angle):
        if angle == 90:
            sample = torch.rot90(sample, k=1, dims=[1, 2])  # 90 degrees along x-y plane
        elif angle == 180:
            sample = torch.rot90(sample, k=2, dims=[1, 2])  # 180 degrees along x-y plane
        elif angle == 270:
            sample = torch.rot90(sample, k=3, dims=[1, 2])  # 270 degrees along x-y plane
        return sample


class SimpleDataset2(Dataset):
    def __init__(self, data):
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            elif data.dtype != torch.float32:
                data = data.float()
        except Exception as e:
            raise ValueError("Data cannot be converted to a torch tensor with dtype torch.float32.") from e
        
        self.m = data
        self.len = self.m.shape[0]

    def __getitem__(self, index):
        sample = self.m[index]
        return sample

    def __len__(self):
        return self.len


class ParallelStaticDataset(Dataset):
    def __init__(self, sample_index, dir_to_database):
        self.sample_index = sample_index
        self.dir_to_database = dir_to_database
        self.M, self.indices = self.load_data_in_parallel()

    def __len__(self):
        return len(self.sample_index)
    
    def _load_data(self, idx):
        perm  = np.load(os.path.join(self.dir_to_database, f'real{idx}_perm_real.npy'))
        return perm, idx

    def load_data_in_parallel(self):
        perm, indices = [], []

        lock = Lock()
        with tqdm(total=len(self.sample_index)) as pbar:
            def task_with_progress(index):
                result = self._load_data(index)
                with lock:
                    pbar.update(1)
                return result
        
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(task_with_progress, self.sample_index))
                results = sorted(results, key=lambda x: x[1])
        
                for reali in results:
                    perm.append(reali[0])
                    indices.append(reali[1])

        # Convert to tensors and preprocess
        perm  = np.array(perm).transpose((0, 1, 4, 2, 3))[:,:,:,4:-4,:]
        nlogk = (np.log10(perm) - np.log10(1.0)) / np.log10(2000)

        indices = np.array(indices)

        data = (torch.tensor(nlogk, dtype=torch.float32), # m
                indices
               )
        return data

    def __getitem__(self, idx):
        return self.M[idx]


class DatasetLSDA(Dataset):
    def __init__(self, sample_index, dir_to_database, timestep, data=None):
        self.sample_index = sample_index
        self.nsample = len(sample_index)
        self.dir_to_database = dir_to_database
        self.timestep = timestep
        m, d = self.load_data_in_parallel() if data is None else data
        self.m = torch.tensor(m, dtype=torch.float32) if not isinstance(m, torch.Tensor) else m
        self.d = torch.tensor(d, dtype=torch.float32) if not isinstance(d, torch.Tensor) else d
        
    def __len__(self):
        return self.nsample
    
    def _load_data(self, idx):
        plume = np.load(os.path.join(self.dir_to_database, f'real{idx}_plume3d.npy'))[self.timestep]
        perm = np.load(os.path.join(self.dir_to_database, f'real{idx}_perm_real.npy'))
        return perm, plume

    def load_data_in_parallel(self):
        perm, plume = [], []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._load_data, self.sample_index))
            for p, pl in tqdm(results):
                perm.append(p)
                plume.append(pl)

        perm = np.array(perm).transpose((0, 1, 4, 2, 3))[:,:,:,4:-4,:]
        plume = np.array(plume)[:,:,:,4:-4,:]        
        nlogk = (np.log10(perm) - np.log10(1.0)) / np.log10(2000)

        return torch.tensor(nlogk, dtype=torch.float32), torch.tensor(plume, dtype=torch.float32)
    
    def __getitem__(self, idx):

        return self.m[idx], self.d[idx]
    

class DatasetLSDA2(Dataset):
    def __init__(self, sample_index, dir_to_database, timestep:int, data=None):
        self.sample_index = sample_index
        self.nsample = len(sample_index)
        self.dir_to_database = dir_to_database
        self.timestep = timestep
        m, d, t = self.load_data_in_parallel() if data is None else data
        self.m = torch.tensor(m, dtype=torch.float32) if not isinstance(m, torch.Tensor) else m
        self.d = torch.tensor(d, dtype=torch.float32) if not isinstance(d, torch.Tensor) else d
        self.t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
        
    def __len__(self):
        return self.nsample
    
    def _load_data(self, idx):
        plume = np.load(os.path.join(self.dir_to_database, f'real{idx}_plume3d.npy'))[[self.timestep]]
        perm = np.load(os.path.join(self.dir_to_database, f'real{idx}_perm_real.npy'))
        return perm, plume

    def load_data_in_parallel(self, well_loc=(99, 255, 0)):
        perm, plume = [], []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_data, idx) for idx in self.sample_index]
            for future in tqdm(as_completed(futures), total=self.nsample, desc="Loading data"):
                p, pl = future.result()
                perm.append(p)
                plume.append(pl)

        perm = np.array(perm).transpose((0, 1, 4, 2, 3))[:,:,:,4:-4,:]
        plume = np.array(plume)[:,:,:,4:-4,:]        
        nlogk = (np.log10(perm) - np.log10(1.0)) / np.log10(2000)

        # Create tstep array filled with zeros
        nx, ny, nz = plume.shape[2], plume.shape[3], plume.shape[4]
        tstep = np.zeros((self.nsample, 1, nx, ny, nz), dtype=np.float32)

        # Set the specific location to the timestep value
        locx, locy, locz = well_loc
        tstep[:, :, locx, locy, locz] = self.timestep
        return (
            torch.tensor(nlogk, dtype=torch.float32),
            torch.tensor(plume, dtype=torch.float32),
            torch.tensor(tstep, dtype=torch.float32)
        )
    
    def __getitem__(self, idx):
        return self.m[idx], self.d[idx], self.t[idx]
