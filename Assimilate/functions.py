import torch.nn as nn
import numpy as np
import torch
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


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


def initialize_prior_latents(data: torch.Tensor, indices_repeat, encoder, decoder, device, batch_size=1):
    num_h_latents = len(decoder.blk_idx)
    m_prior_list, z_latent_list = [], []
    h_latent_list = [[] for i in range(num_h_latents)]
    _indices_repeat = indices_repeat.unsqueeze(0).unsqueeze(0).to(device)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        z, h = encoder(torch.tensor(batch, dtype=torch.float32).to(device))
        m_prior_batch = decoder(z, h, indices_repeat=_indices_repeat)[0][0].detach().cpu()
        m_prior_list.append(m_prior_batch)
        
        z_latent_list.append(z.detach().cpu())
        for j in range(num_h_latents):
            h_latent_list[j].append(h[j].detach().cpu())

    latent_z = torch.cat(z_latent_list, dim=0)
    latent_h = [torch.cat(_h_collect, dim=0) for _h_collect in h_latent_list]
    latents = [latent_z] + latent_h[::-1]
    m_prior = torch.cat(m_prior_list, dim=0).numpy()
    return m_prior, latents


def sample_mvnormal(*, C_d, rng: np.random._generator.Generator, size: int,):
    # Standard normal samples
    z = rng.standard_normal(size=(size, C_d.shape[0]))

    # A 2D covariance matrix was passed
    if C_d.ndim == 2:
        return z @ C_d 

    # A 1D diagonal of a covariance matrix was passed
    else:
        return z @ np.diag(C_d)


def get_realizations(m_data, ensemble_size):
    start = 0
    end = m_data.shape[0]
    # Generate a set of unique random integers
    random_indices = random.sample(range(start, end + 1), ensemble_size)
    return m_data[[random_indices[0]]], m_data[random_indices[1:]], random_indices


def generate_index_from_coarse_to_fine(scale_index_coarse2fine, 
                                       density_coarse2fine, 
                                       indices_shape_coarse2fine, 
                                       indices_ratio_coarse2fine):

    """
    Generate index from fine to coarse.
    scale_index_coarse2fine: List of int. 
        For example: [1, 2, 4]

    density_coarse2fine: List of torch.tensor. 
        For example: [torch.Size([1, 1, 4, 4, 1]), torch.Size([1, 1, 8, 8, 1]), torch.Size([1, 1, 32, 32, 1])]
    
    indices_shape_coarse2fine: List of tuple. 
        For example: [(4, 4, 1), (8, 8, 1), (32, 32, 1)]
        
    indices_ratio_coarse2fine: List of float. 
        For example: [0.3, 0.3, 1.0]
    """
    def _repeat_interleave(index_map, scale_factor):
        """
        Repeat and interleave the index map.
        index_map: torch.tensor. Shape: (H, W, D)
        scale_factor: tuple. 
        """
        for _dim, _scale in enumerate(scale_factor):
            index_map = index_map.repeat_interleave(_scale, dim=_dim)
        return index_map 

    assert len(scale_index_coarse2fine) == len(density_coarse2fine) == len(indices_shape_coarse2fine) == len(indices_ratio_coarse2fine)
    
    indices_ratio_coarse2fine[-1] = 1.0
    selected_masks = []
    prev_shape = indices_shape_coarse2fine[0] # (h, w, d)
    indices_repeat = scale_index_coarse2fine[-1] * torch.ones(prev_shape) # (h, w, d)
    
    for curr_scale, curr_density, curr_shape, curr_ratio in zip(
        scale_index_coarse2fine, density_coarse2fine, indices_shape_coarse2fine, indices_ratio_coarse2fine):
    
        scale_factor = [(a//b) for (a, b) in zip(curr_shape, prev_shape)] # e.g., (2, 2, 1)
        indices_repeat = _repeat_interleave(indices_repeat, scale_factor) # e.g., (2h, 2w, 2d)

        num_topk_values = int(np.prod(curr_shape)*curr_ratio)
        topk_values, topk_indices = torch.topk(-curr_density.view(-1), num_topk_values)
        mask = torch.zeros_like(curr_density.view(-1))
        mask[topk_indices] = 1 # curr_scale
        mask = mask.view(curr_shape) 
        
        update_condition = (mask == 1) & (indices_repeat > curr_scale - 1)
        indices_repeat[update_condition] = curr_scale
        selected_masks.append(update_condition)
        prev_shape = curr_shape

    return indices_repeat, selected_masks


def generate_selected_density(density3D, selected_indices_shape):
    selected_density = []
    density_shape = tuple(density3D.shape)
    density_map_5D = torch.tensor(density3D, dtype=torch.float32)
    density_map_5D = density_map_5D.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, H, W, D]
    for _selected_scale in selected_indices_shape:
        _kernel_size = tuple([a//b for (a, b) in zip(list(density_shape), list(_selected_scale))])
        PoolOperator = nn.AvgPool3d(kernel_size=_kernel_size, stride=_kernel_size)
        selected_density.append(PoolOperator(density_map_5D))
    return selected_density


def get_restore_locations(latents: List[torch.Tensor], selected_masks, selected_scale_index):

    nens, nchannel = latents[0].shape[0], latents[0].shape[1]
    unmasked_latents, unmasked_locations = [], []
    for idx, mask0 in zip(selected_scale_index, selected_masks):
        z0 = latents[idx].view(nens, nchannel, -1)
        elements, locations = extract_unmasked_elements(z0, mask0)
        unmasked_latents.append(elements), unmasked_locations.append(locations)
        print(f"Level {idx}: All--{z0.shape}. Unmasked--{elements.shape}. Location--{locations.shape}")
        
    ensemble = torch.cat(unmasked_latents, dim=2).view(nens, -1).detach().cpu().numpy()

    return ensemble, unmasked_locations


def extract_unmasked_elements(z, mask):
    """
    Extract unmasked elements from z based on the mask and return them as a single vector,
    along with the locations of these unmasked elements considering the broadcasting of mask.
    
    Parameters:
    z (torch.Tensor): The latent variable tensor of shape (nens, channel, spatial_dims).
    mask (torch.Tensor): A binary mask tensor of shape (1, 1, spatial_dims).
    
    Returns:
    unmasked_elements (torch.Tensor): A 1D tensor containing the unmasked elements of z.
    locations (torch.Tensor): A 2D tensor where each row represents the location of an unmasked element in z.
    """
    # Ensure mask and z are flatten in the spatial dimension
    mask = mask.view(1, 1, -1)
    z = z.view(z.size(0), z.size(1), -1)
    
    # Ensure mask is boolean and broadcast it to match z's shape
    if not isinstance(z, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensors.")

    mask_bool = mask.bool().expand_as(z)
    
    # Extract unmasked elements and reshape to (nens, channel, num_unmasked)
    unmasked_elements = torch.masked_select(z, mask_bool).view(z.size(0), z.size(1), -1)
    
    # Find the locations of unmasked elements
    locations = torch.nonzero(mask, as_tuple=False)[:,-1]
    
    return unmasked_elements, locations


def restore_unmasked_elements(unmasked_elements, locations, z):
    """
    Restore unmasked elements to their original locations within a tensor 'z'.

    Parameters:
    unmasked_elements (torch.Tensor): A 1D tensor containing the unmasked elements.
    locations (torch.Tensor): A 1D tensor indicating the original locations of unmasked elements in 'z'.
    z (torch.Tensor): The tensor from which elements were originally extracted, used to determine the shape of the restored tensor.

    Returns:
    restored_tensor (torch.Tensor): A tensor with the same shape as 'z', where unmasked elements have been restored to their original locations, and the rest of the elements are zeros.
    """

    restored_tensor = z.clone().detach().cpu() # restored_tensor = torch.zeros_like(z, dtype=z.dtype)
    restored_tensor[:,:,locations] = unmasked_elements # .shape    

    return restored_tensor

 
