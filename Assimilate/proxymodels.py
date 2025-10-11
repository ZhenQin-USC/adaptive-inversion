import os
import torch.nn as nn
import numpy as np
import torch
import json
from functools import reduce
from operator import mul
from os.path import (join, isfile)
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
from .functions import generate_index_from_coarse_to_fine, get_restore_locations, restore_unmasked_elements
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from mpi4py import MPI
import multiprocessing as mp

save_fig = True # False # 


def multiply_scale_factors(scale_factors):
    # Use reduce to apply multiplication over each corresponding element in the tuples
    return tuple(reduce(mul, factors) for factors in zip(*scale_factors))


def calculate_indices_shapes(image_size, decode_factors, all_grain_factors):
    # Step 1: Calculate the highest level using decode_factors
    highest_level = tuple(image_size[i] // decode_factors[i] for i in range(len(image_size)))

    # Step 2: Calculate subsequent levels using all_grain_factors
    levels = [highest_level]
    for factor in all_grain_factors:
        next_level = tuple(highest_level[i] // factor[i] for i in range(len(highest_level)))
        levels.append(next_level)
    
    # Step 3: Reverse the levels list and concatenate
    indices_shapes = levels[::-1]
    
    return indices_shapes # [(2, 2, 1), (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1)])


def generate_selected_density(density, selected_indices_shape):
    selected_density = []
    density_shape = tuple(density.shape)
    density_map_5D = torch.tensor(density, dtype=torch.float32)
    density_map_5D = density_map_5D.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, H, W, D]
    for _selected_scale in selected_indices_shape:
        _kernel_size = tuple([a//b for (a, b) in zip(list(density_shape), list(_selected_scale))])
        PoolOperator = nn.AvgPool3d(kernel_size=_kernel_size, stride=_kernel_size)
        selected_density.append(PoolOperator(density_map_5D))
    return selected_density


def load_modules(checkpoint_dir: str, Model: Callable, device) -> Tuple:
    with open(join(checkpoint_dir, 'model_config.json')) as json_file:
        model_config = json.load(json_file)
        model_config['scale_factors'] = [tuple(_) for _ in model_config['scale_factors']]
        model_config['index_sampling_type'] = 'bottom2top'

    with open(join(checkpoint_dir, 'train_config.json')) as json_file:
        train_config = json.load(json_file)

    with open(join(checkpoint_dir, 'disc_config.json')) as json_file:
        disc_config = json.load(json_file)

    autoencoder = Model(model_config, disc_config, train_config).to(device) 
    autoencoder._load_model(checkpoint_dir) # checkpoint = torch.load(join(checkpoint_dir, 'checkpoint.pt'))
    encoder = autoencoder._model.encoder
    decoder = autoencoder._model.decoder
    encoder.eval(), decoder.eval()
    return autoencoder, encoder, decoder


def initialize_indices_and_masks(image_size, scale_factors, decode_factors, 
                                 selected_scale_index, selected_indices_ratio, density3D):
    """
    Initialize scale factors, indices shapes, densities, and masks for the decoder.

    :param decoder: The decoder object containing masker attributes.
    :param image_size: The size of the image.
    :param selected_scale_index: Indices of selected scales.
    :param selected_indices_ratio: Ratios of selected indices.
    :param density3D: The 3D density configuration.
    :return: A dictionary with initialized scale factors, indices shapes, densities, and masks.
    """

    # Extract scale and decode factors from decoder
    # scale_factors = decoder.masker.all_grain_factors
    # decode_factors = decoder.masker.decode_factors[0]

    # Calculate indices shapes
    indices_shapes = calculate_indices_shapes(image_size, decode_factors, scale_factors)
    
    # Select indices shapes based on given scale indices
    selected_indices_shape = [indices_shapes[i] for i in selected_scale_index]

    # Generate selected density based on provided 3D density and selected indices shape
    selected_density = generate_selected_density(density3D, selected_indices_shape)

    # Calculate number of scales
    num_scales = len(indices_shapes)

    # Generate indices repeat and selected masks
    indices_repeat, selected_masks = generate_index_from_coarse_to_fine(
        scale_index_coarse2fine=selected_scale_index,
        density_coarse2fine=selected_density,
        indices_shape_coarse2fine=selected_indices_shape,
        indices_ratio_coarse2fine=selected_indices_ratio
    )
    # return indices_repeat, selected_masks, indices_shapes
    # Return all initialized values in a dictionary
    return {
        'scale_factors': scale_factors,
        'decode_factors': decode_factors,
        'indices_shapes': indices_shapes,
        'selected_indices_shape': selected_indices_shape,
        'selected_density': selected_density,
        'num_scales': num_scales,
        'indices_repeat': indices_repeat,
        'selected_masks': selected_masks
    }

# TO DO: General parallelization helper using MPI with max_workers argument
def run_parallel_tasks_mpi(task_func: Callable, task_range: range, *args: Any, desc: str = "Processing tasks", max_workers=None):
    # MPI setup inside the function
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Set max_workers to size if not specified
    if max_workers is None or max_workers > size:
        max_workers = size
    print("max_workers", max_workers)

    results = []

    # If this rank is not within the max_workers range, it will be idle
    if rank < max_workers:
        # Divide the tasks among active ranks (up to max_workers)
        chunk_size = len(task_range) // max_workers
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size if rank != max_workers - 1 else len(task_range)
        
        # Each process handles its part of the task range
        local_task_range = task_range[start_idx:end_idx]
        
        local_results = []
        for i in tqdm(local_task_range, desc=f"Rank {rank} {desc}"):
            local_results.append(task_func(i, *args))

        # Gather results at the root process
        all_results = comm.gather(local_results, root=0)
    else:
        # Idle ranks participate in the gather operation but send nothing
        all_results = comm.gather([], root=0)

    # Combine results at the root process
    if rank == 0:
        # Flatten the list of lists
        results = [item for sublist in all_results for item in sublist]
    
    return results

# General parallelization helper
def run_parallel_tasks(task_func: Callable, task_range: range, *args: Any, desc: str = "Processing tasks", max_workers=4):
    results = []
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(task_func, i, *args) for i in task_range]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers to match your CPU capacity
        futures = [executor.submit(task_func, i, *args) for i in task_range]

        # Track progress with tqdm
        for future in tqdm(futures, desc=desc):
            results.append(future.result())

    return results

# def run_parallel_tasks_mp(task_func: Callable, task_range: range, *args: Any, desc: str = "Processing tasks", max_workers=None):
#     # Set max_workers to the number of CPU cores if not provided
#     if max_workers is None:
#         # max_workers = mp.cpu_count()
#         max_workers = min(mp.cpu_count(), 4)  # Use at most 4 processes, or fewer if the system has fewer CPU cores

#     # Initialize a pool of worker processes
#     with mp.Pool(processes=max_workers) as pool:
#         # Prepare arguments for each task (passing the function and the corresponding parameters)
#         tasks = [(i, *args) for i in task_range]

#         # Use starmap to apply task_func to each set of arguments, using tqdm for progress
#         results = list(tqdm(pool.starmap(task_func, tasks), desc=desc, total=len(task_range)))

#     return results

# Parallel task helper using multiprocessing
def run_parallel_tasks_mp(task_func: Callable, task_range: range, *args: Any, desc: str = "Processing tasks", max_workers=None):
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit number of workers

    # Initialize a pool of worker processes
    with mp.Pool(processes=max_workers) as pool:
        tasks = [(i, *args) for i in task_range]

        # Use imap_unordered for dynamic progress tracking
        results = []
        for result in tqdm(pool.imap_unordered(task_func_wrapper, tasks), total=len(task_range), desc=desc):
            results.append(result)

    return results

# Wrapper for task function to handle tuple unpacking
def task_func_wrapper(args):
    task_func, *params = args
    return task_func(*params)

# Process batch for generating latent variables using encoder and decoder
def process_batch_latents(start_idx: int, m: torch.Tensor, batch_size: int, encoder, decoder, device, _indices_repeat):
    batch = m[start_idx:start_idx + batch_size].clone().detach().to(device)
    h = encoder(batch)
    m_batch = decoder(h, indices_repeat=_indices_repeat)[0][0].detach().cpu()

    latent_list = [[] for _ in range(len(h))]
    for j, _h in enumerate(h):
        latent_list[j].append(_h.detach().cpu())
    
    return m_batch, latent_list

# Process batch for decoding using decoder only
def process_batch_decode(start_idx: int, hs_ens: List[torch.Tensor], batch_size: int, decoder, device, _indices_repeat):
    h_batch = [h[start_idx:start_idx + batch_size].to(device) for h in hs_ens]
    m_batch = decoder(h_batch, indices_repeat=_indices_repeat)[0][0].detach().cpu()
    return m_batch


class Proxy:
    def __init__(self, 
                 setup: Dict, 
                 checkpoint_dir: str, 
                 Model: Callable, 
                 simulator: Callable, 
                 **kwargs):
        
        self.device = setup['device']
        self.device = torch.device(setup['device']) if isinstance(setup['device'], str) else setup['device']
        density3D = setup.get('measurement_density')
        self.image_size = setup.get('image_size')
        self.selected_scale_index = setup.get('selected_scale_index') # [1, 2, 4, 5]
        self.selected_indices_ratio = setup.get('selected_indices_ratio') # [0.4, 0.6, 0.82, 1.0]
        self.batch_size = kwargs.get('batch_size', 1)
        self.dir_to_project = kwargs.get('dir_to_project', None)
        _, self.encoder, self.decoder = load_modules(checkpoint_dir, Model, self.device)
        self.simulator = simulator

        self.posteriors, self.d_posteriors, self.latents = [], [], []

        self.scale_factors = self.decoder.masker.all_grain_factors
        self.decode_factors = multiply_scale_factors(self.decoder.masker.decode_factors)

        
        outputs_dict = initialize_indices_and_masks(
            self.image_size, self.scale_factors, self.decode_factors, 
            self.selected_scale_index, self.selected_indices_ratio, density3D)
        self.indices_repeat = outputs_dict["indices_repeat"]
        self.selected_masks = outputs_dict["selected_masks"]
        self.indices_shapes = outputs_dict["indices_shapes"]
        # self.indices_shapes = calculate_indices_shapes(self.image_size, self.decode_factors, self.scale_factors)
        # self.selected_indices_shape = [self.indices_shapes[i] for i in self.selected_scale_index]
        # self.selected_density = generate_selected_density(density3D, self.selected_indices_shape)
        # self.num_scales = len(self.indices_shapes)
        # self.indices_repeat, self.selected_masks = generate_index_from_coarse_to_fine(
        #     scale_index_coarse2fine=self.selected_scale_index,
        #     density_coarse2fine=self.selected_density,
        #     indices_shape_coarse2fine=self.selected_indices_shape,
        #     indices_ratio_coarse2fine=self.selected_indices_ratio
        # )

    def __call__(self, ensemble: np.ndarray, iteration: int=None) -> np.ndarray:

        # Update the latent variable with the ensemble and decode
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format

        # Run the simulation to generate simulated outputs
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens, iteration=iteration)

        # Store the posterior model and observable data
        self.posteriors.append(m_ens)
        self.d_posteriors.append(d_ens)
        self.latents.append(new_latents)
        return d_ens

    def get_posteriors(self, ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens)
        return m_ens, d_ens

    def initialize(self, m_true: torch.Tensor):
        self.m_prior, latents, self.init_ensemble, self.unmasked_locations \
            = self.generate_latent_variables(m_true)
        self.latents.append(latents.copy())
        self.nchannel = latents[0].shape[1]
        self.split_sizes = [_.numel() for _ in self.unmasked_locations]

    def generate_latent_variables(self, m: torch.Tensor) -> Tuple[np.ndarray, List[torch.Tensor]]:
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m)

        m_list, latent_list = [], None
        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)

        for i in range(0, len(m), self.batch_size):
            batch = m[i:i+self.batch_size].clone().detach().to(self.device)
            h = self.encoder(batch)
            m_batch = self.decoder(h, indices_repeat=_indices_repeat)[0][0].detach().cpu()
            m_list.append(m_batch)

            if latent_list is None:
                latent_list = [[] for _ in range(len(h))]

            for j, _h in enumerate(h):
                latent_list[j].append(_h.detach().cpu())

        reversed_latents = [torch.cat(_h_collect, dim=0) for _h_collect in latent_list]
        m_recon = torch.cat(m_list, dim=0).numpy()
        latents = reversed_latents[::-1]
        x_ensemble, unmasked_locations = get_restore_locations(
            latents, self.selected_masks, self.selected_scale_index)

        return m_recon, latents, x_ensemble, unmasked_locations

    def _decode(self, latents: List[torch.Tensor]) -> np.ndarray:
        hs_ens = latents[::-1]  # Reverse to maintain order
        nsample = hs_ens[0].shape[0]
        m_ens = []
        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)

        for i in range(0, nsample, self.batch_size):
            h_batch = [_[i:i+self.batch_size].to(self.device) for _ in hs_ens]
            m_list = self.decoder(h_batch, indices_repeat=_indices_repeat)[0][0].detach().cpu()
            m_ens.append(m_list)

        return torch.stack(m_ens, dim=0).numpy()

    def _ensemble_to_latents(self, ensemble: np.ndarray) -> List[torch.Tensor]:
        nens, nchannel = ensemble.shape[0], self.nchannel
        latents = self.latents[-1]

        ensemble_tensor = torch.tensor(ensemble, dtype=torch.float32).view(nens, nchannel, -1)
        splited_latents = torch.split(ensemble_tensor, self.split_sizes, dim=2)

        new_latents = [_.clone() for _ in latents]
        for _scale_idx, _location, _latent in zip(self.selected_scale_index, self.unmasked_locations, splited_latents):
            _new_latent = restore_unmasked_elements(_latent, _location, latents[_scale_idx].view(nens, nchannel, -1))
            new_latents[_scale_idx] = _new_latent.view(latents[_scale_idx].size())

        return new_latents


class Proxy2:
    def __init__(self, 
                 encoder, decoder, device, simulator, 
                 indices_repeat, selected_masks, selected_scale_index, 
                 **kwargs):
        
        self.device = device
        self.batch_size = kwargs.get('batch_size', 1)
        self.dir_to_project = kwargs.get('dir_to_project', None)
        self.encoder, self.decoder = encoder, decoder
        self.simulator = simulator
        self.selected_scale_index = selected_scale_index
        self.indices_repeat, self.selected_masks = indices_repeat, selected_masks
        self.posteriors, self.d_posteriors, self.latents = [], [], []

    def __call__(self, ensemble: np.ndarray, iteration: int=None) -> np.ndarray:

        # Update the latent variable with the ensemble and decode
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format

        # Run the simulation to generate simulated outputs
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens, iteration=iteration)

        # Store the posterior model and observable data
        self.posteriors.append(m_ens)
        self.d_posteriors.append(d_ens)
        self.latents.append(new_latents)
        return d_ens

    def get_posteriors(self, ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens)
        return m_ens, d_ens

    def initialize(self, m_true: torch.Tensor):
        self.m_prior, latents, self.init_ensemble, self.unmasked_locations \
            = self.generate_latent_variables(m_true)
        self.latents.append(latents.copy())
        self.nchannel = latents[0].shape[1]
        self.split_sizes = [_.numel() for _ in self.unmasked_locations]

    def generate_latent_variables(self, m: torch.Tensor) -> Tuple[np.ndarray, List[torch.Tensor]]:
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m)

        m_list, latent_list = [], None
        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)

        for i in range(0, len(m), self.batch_size):
            batch = m[i:i+self.batch_size].clone().detach().to(self.device)
            h = self.encoder(batch)
            m_batch = self.decoder(h, indices_repeat=_indices_repeat)[0][0].detach().cpu()
            m_list.append(m_batch)

            if latent_list is None:
                latent_list = [[] for _ in range(len(h))]

            for j, _h in enumerate(h):
                latent_list[j].append(_h.detach().cpu())

        reversed_latents = [torch.cat(_h_collect, dim=0) for _h_collect in latent_list]
        m_recon = torch.cat(m_list, dim=0).numpy()
        latents = reversed_latents[::-1]
        x_ensemble, unmasked_locations = get_restore_locations(
            latents, self.selected_masks, self.selected_scale_index)

        return m_recon, latents, x_ensemble, unmasked_locations

    def _decode(self, latents: List[torch.Tensor]) -> np.ndarray:
        hs_ens = latents[::-1]  # Reverse to maintain order
        nsample = hs_ens[0].shape[0]
        m_ens = []
        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)

        for i in range(0, nsample, self.batch_size):
            h_batch = [_[i:i+self.batch_size].to(self.device) for _ in hs_ens]
            m_list = self.decoder(h_batch, indices_repeat=_indices_repeat)[0][0].detach().cpu()
            m_ens.append(m_list)

        return torch.vstack(m_ens).numpy()

    def _ensemble_to_latents(self, ensemble: np.ndarray) -> List[torch.Tensor]:
        nens, nchannel = ensemble.shape[0], self.nchannel
        latents = self.latents[-1]

        ensemble_tensor = torch.tensor(ensemble, dtype=torch.float32).view(nens, nchannel, -1)
        splited_latents = torch.split(ensemble_tensor, self.split_sizes, dim=2)

        new_latents = [_.clone() for _ in latents]
        for _scale_idx, _location, _latent in zip(self.selected_scale_index, self.unmasked_locations, splited_latents):
            _new_latent = restore_unmasked_elements(_latent, _location, latents[_scale_idx].view(nens, nchannel, -1))
            new_latents[_scale_idx] = _new_latent.view(latents[_scale_idx].size())

        return new_latents


class Proxy3:
    def __init__(self, 
                 encoder, decoder, device, simulator, 
                 indices_repeat, selected_masks, selected_scale_index, 
                 **kwargs):
        
        self.device = device
        self.batch_size = kwargs.get('batch_size', 1)
        self.dir_to_project = kwargs.get('dir_to_project', None)
        self.encoder, self.decoder = encoder, decoder
        self.simulator = simulator
        self.selected_scale_index = selected_scale_index
        self.indices_repeat, self.selected_masks = indices_repeat, selected_masks
        self.posteriors, self.d_posteriors, self.latents = [], [], []

    def __call__(self, ensemble: np.ndarray, iteration: int=None) -> np.ndarray:

        # Update the latent variable with the ensemble and decode
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format

        # Run the simulation to generate simulated outputs
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens, iteration=iteration)

        # Store the posterior model and observable data
        self.posteriors.append(m_ens)
        self.d_posteriors.append(d_ens)
        self.latents.append(new_latents)
        return d_ens

    def get_posteriors(self, ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_latents = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format
        m_ens = self._decode(new_latents)
        d_ens = self.simulator(m_ens)
        return m_ens, d_ens

    def initialize(self, m_true: torch.Tensor):
        self.m_prior, latents, self.init_ensemble, self.unmasked_locations \
            = self.generate_latent_variables(m_true)
        self.latents.append(latents.copy())
        self.nchannel = latents[0].shape[1]
        self.split_sizes = [_.numel() for _ in self.unmasked_locations]

    # Updated generate_latent_variables function
    def generate_latent_variables(self, m: torch.Tensor) -> Tuple[np.ndarray, List[torch.Tensor]]:
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m)

        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)
        nsample = len(m)

        # Use the same run_parallel_tasks helper
        task_range = range(0, nsample, self.batch_size)
        # Call the MPI-based parallelization helper
        results = run_parallel_tasks(process_batch_latents, task_range, m, self.batch_size, 
                                     self.encoder, self.decoder, self.device, _indices_repeat, 
                                     desc="Processing batches", max_workers = os.cpu_count())
        # results = run_parallel_tasks_mp(process_batch_latents, task_range, m, self.batch_size, 
        #                                 self.encoder, self.decoder, self.device, _indices_repeat, 
        #                                 desc="Processing batches")
        
        # Unpack results: Process results to get m_ens and latent_list
        m_ens = [m_batch for m_batch, _ in results]
        latent_list = [[] for _ in range(len(results[0][1]))]
        for _, batch_latent in results:
            for j, _h_collect in enumerate(batch_latent):
                latent_list[j].extend(_h_collect)

        # Reverse and concatenate the latent variables
        reversed_latents = [torch.cat(_h_collect, dim=0) for _h_collect in latent_list]
        m_recon = torch.cat(m_ens, dim=0).numpy()
        latents = reversed_latents[::-1]
        
        # Restore x_ensemble and unmasked_locations
        x_ensemble, unmasked_locations = get_restore_locations(
            latents, self.selected_masks, self.selected_scale_index)

        return m_recon, latents, x_ensemble, unmasked_locations

    # Updated _decode function
    def _decode(self, latents: List[torch.Tensor]) -> np.ndarray:
        hs_ens = latents[::-1]  # Reverse to maintain order
        nsample = hs_ens[0].shape[0]
        _indices_repeat = self.indices_repeat.unsqueeze(0).unsqueeze(0).to(self.device)

        task_range = range(0, nsample, self.batch_size)
        # Call the parallelization helper for decoding
        m_ens = run_parallel_tasks(process_batch_decode, task_range, hs_ens, self.batch_size, 
                                   self.decoder, self.device, _indices_repeat, 
                                   desc="Decoding batches", max_workers = os.cpu_count())
        # m_ens = run_parallel_tasks_mp(process_batch_decode, task_range, hs_ens, self.batch_size, 
        #                               self.decoder, self.device, self.indices_repeat,
        #                               desc="Decoding batches")

        # Concatenate all the decoded batches
        return torch.vstack(m_ens).numpy()

    def _ensemble_to_latents(self, ensemble: np.ndarray) -> List[torch.Tensor]:
        nens, nchannel = ensemble.shape[0], self.nchannel
        latents = self.latents[-1]

        ensemble_tensor = torch.tensor(ensemble, dtype=torch.float32).view(nens, nchannel, -1)
        splited_latents = torch.split(ensemble_tensor, self.split_sizes, dim=2)

        new_latents = [_.clone() for _ in latents]
        for _scale_idx, _location, _latent in zip(self.selected_scale_index, self.unmasked_locations, splited_latents):
            _new_latent = restore_unmasked_elements(_latent, _location, latents[_scale_idx].view(nens, nchannel, -1))
            new_latents[_scale_idx] = _new_latent.view(latents[_scale_idx].size())

        return new_latents


