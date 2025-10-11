import gc
import pickle
import torch
import lmdb
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import psutil
import random
import os
import json
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from os.path import join
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock


class ParallelDatasetSimpleProxy(Dataset): # (512, 128, 1) - 2D reservoir
    def __init__(self, sample_index, dir_to_database, timestep, pred_length=8):
        self.sample_index = sample_index
        self.dir_to_database = dir_to_database
        self.timestep = timestep
        self.pred_length = pred_length
        self.total_step = len(timestep)

        # create a dictionary of item keys
        self.precomputed_keys = {}
        self._precompute_keys()
        self.nsample = len(self.precomputed_keys)

        # Load data in parallel
        self.M, self.S, self.P, self.C, self.indices = self.load_data_in_parallel()
    
    def __len__(self):
        return self.nsample
    
    def _precompute_keys(self):
        pred_length = self.pred_length
        total_step = self.total_step
        idx = 0
        for sample_index, real_index in enumerate(self.sample_index):
            for step_start in range(total_step - pred_length):
                step_index = list(range(step_start, step_start + self.pred_length + 1)) # [initial step + prediction steps]
                # Store the precomputed information in the dictionary
                self.precomputed_keys[idx] = (sample_index, real_index, step_index)
                idx += 1

    def _load_data(self, idx):
        perm  = np.load(os.path.join(self.dir_to_database, f'real{idx}_perm_real.npy'))
        plume = np.load(os.path.join(self.dir_to_database, f'real{idx}_plume3d.npy'))[self.timestep]
        press = np.load(os.path.join(self.dir_to_database, f'real{idx}_press3d.npy'))[self.timestep]
        cntrl = np.load(os.path.join(self.dir_to_database, f'real{idx}_input_control.npy'))[:len(self.timestep[1:]),:]
        return perm, plume, press, cntrl, idx

    def load_data_in_parallel(self):
        perm, plume, press, cntrl, indices = [], [], [], [], []
        
        lock = Lock()
        with tqdm(total=len(self.sample_index)) as pbar:
            def task_with_progress(index):
                result = self._load_data(index)
                with lock:
                    pbar.update(1)
                return result
        
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(task_with_progress, self.sample_index))
                results = sorted(results, key=lambda x: x[4])
        
                for reali in results:
                    _perm, _plume, _press, _cntrl, _index = reali
                    perm.append(_perm)
                    plume.append(_plume)
                    press.append(_press)
                    cntrl.append(_cntrl)
                    indices.append(_index)
                
        # Convert to tensors and preprocess
        perm  = np.array(perm).transpose((0, 1, 4, 2, 3))[:,:,:,4:-4,:]
        nlogk = (np.log10(perm) - np.log10(1.0)) / np.log10(2000)
        plume = np.array(plume)[:,:,:,4:-4,:]    
        press = (np.array(press)[:,:,:,4:-4,:] - 9810) / (33110 - 9810)
        cntrl = np.array(cntrl)
        indices = np.array(indices)
        
        data = (torch.tensor(nlogk, dtype=torch.float32), # m
                torch.tensor(plume, dtype=torch.float32), # s
                torch.tensor(press, dtype=torch.float32), # p
                torch.tensor(cntrl, dtype=torch.float32), # c
                indices
               )
        return data
    
    def __getitem__(self, idx):
        sample_index, real_index, step_index = self.precomputed_keys[idx]
                
        s0 = self.S[sample_index, step_index[0:1]]
        p0 = self.P[sample_index, step_index[0:1]]
        st = self.S[sample_index, step_index[1:]].unsqueeze(dim=1)
        pt = self.P[sample_index, step_index[1:]].unsqueeze(dim=1)

        static = self.M[sample_index]
        states = torch.cat((p0, s0), dim=0)
        contrl = torch.zeros(st.size(), dtype=torch.float32)
        contrl[:, :, 100, 255, 0] = 1
        output = torch.cat((pt, st), dim=1)
        return (contrl, states, static), output
    
