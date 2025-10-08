import numpy as np
import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple


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


def generate_index_from_coarse_to_fine(scale_index_coarse2fine, 
                                       density_coarse2fine, 
                                       indices_shape_coarse2fine, 
                                       indices_ratio_coarse2fine):

    """
    Generate index from fine to coarse.
    
    Parameters:
        scale_index_coarse2fine: List of int
            A list of scale indices from coarse to fine resolution.
            Example: [1, 2, 4] (where 1 is the coarsest and 4 is the finest).
    
        density_coarse2fine: List of torch.Tensor
            A list of tensors representing density values at different resolutions.
            Each tensor has a shape corresponding to the grid resolution at that scale.
            Example shapes: [torch.Size([1, 1, 4, 4, 1]), torch.Size([1, 1, 8, 8, 1]), torch.Size([1, 1, 32, 32, 1])].
    
        indices_shape_coarse2fine: List of tuple
            A list of tuples representing the spatial shape of each resolution level.
            Example: [(4, 4, 1), (8, 8, 1), (32, 32, 1)].
    
        indices_ratio_coarse2fine: List of float
            A list of float values specifying the proportion of grid points selected at each resolution.
            Example: [0.3, 0.3, 1.0] (30% of points are selected at the first two scales, and all points are selected at the finest scale).
    
    Returns:
        scale_indices: torch.Tensor
            A tensor of shape `indices_shape_coarse2fine[-1]` (i.e., the finest resolution grid),
            where each element represents the coarsest scale index at which that grid point was selected.
            Example: A tensor of shape (32, 32, 1) with values from {1, 2, 4}, indicating whether a point
            was selected at scale 1 (coarsest), scale 2, or scale 4 (finest).
    
        selected_mask: List of torch.Tensor
            A list of boolean tensors, one for each scale, where `True` values indicate the selected grid points
            at that resolution.
            Each tensor has the same shape as `density_coarse2fine[i]`, corresponding to the grid at that scale.
            Example: A list of tensors with shapes [(4, 4, 1), (8, 8, 1), (32, 32, 1)] where `True` values
            mark the selected indices at each resolution.
    """

    def _repeat_interleave(index_map, scale_factor):
        """
        Repeat and interleave the index map across spatial dimensions.
        index_map: torch.Tensor. Shape: (H, W, D)
        scale_factor: tuple of ints, indicating upsampling factors for each dimension.
        """
        for _dim, _scale in enumerate(scale_factor):
            index_map = index_map.repeat_interleave(_scale, dim=_dim)
        return index_map 

    assert len(scale_index_coarse2fine) == len(density_coarse2fine) == len(indices_shape_coarse2fine) == len(indices_ratio_coarse2fine)
    
    indices_ratio_coarse2fine[-1] = 1.0  # Ensure the finest level selects all points
    selected_mask = []  # Stores masks for all levels
    prev_shape = indices_shape_coarse2fine[0]  # Initial spatial shape (h, w, d)
    scale_indices = scale_index_coarse2fine[-1] * torch.ones(prev_shape)  # Initialize with finest scale index
    
    for curr_scale, curr_density, curr_shape, curr_ratio in zip(
        scale_index_coarse2fine, density_coarse2fine, indices_shape_coarse2fine, indices_ratio_coarse2fine):
    
        # Compute the scale factor for upsampling
        scale_factor = [(a // b) for (a, b) in zip(curr_shape, prev_shape)]
        scale_indices = _repeat_interleave(scale_indices, scale_factor)

        # Determine the number of selected points
        num_topk_values = int(np.prod(curr_shape) * curr_ratio)
        topk_values, topk_indices = torch.topk(-curr_density.view(-1), num_topk_values)
        # Generate the selection mask
        mask = torch.zeros_like(curr_density.view(-1))
        mask[topk_indices] = 1 # curr_scale
        mask = mask.view(curr_shape)

        # Expand mask shape to (1, 1, nx, ny, nz)
        expanded_mask = mask.unsqueeze(0).unsqueeze(0)

        # Store the mask for this resolution

        # Update scale_indices where this scale is selected
        update_condition = (mask == 1) & (scale_indices > curr_scale - 1)
        scale_indices[update_condition] = curr_scale
        selected_mask.append(update_condition.unsqueeze(0).unsqueeze(0))

        # Update previous shape reference
        prev_shape = curr_shape

    return scale_indices, selected_mask


class MultiResolutionLatentSelector(nn.Module):
    def __init__(self, 
                 all_grain_factors, 
                 all_latent_shapes,
                 upsampling_mode='nearest'):
        """
        Multi-Resolution Latent Variable Selector.
        This module selects specific elements from multi-resolution latent variables based on boolean masks
        and restores them back to their original 5D shape.

        all_latent_shapes (List[Tuple[int]]): List of original shapes for each latent variable.
        """
        super().__init__()
        self.all_grain_factors = [tuple(_) for _ in all_grain_factors]
        self.all_latent_shapes = [tuple(_) for _ in all_latent_shapes]
        self.latent_grains_shapes = [tuple(list(_)[2:]) for _ in all_latent_shapes[::-1]]
        self.upsampling_mode = upsampling_mode
        self.scale_indices, self.selected_mask, self.selected_density_map = None, None, None
        self.selected_locations = []  # Store selected locations for each scale

    def generate_mask(self, 
                      density, 
                      selected_scale_index, 
                      selected_grain_ratio):
        
        self.selected_density_map = generate_selected_density(
            density, self.latent_grains_shapes)

        self.scale_indices, selected_mask = generate_index_from_coarse_to_fine(
            scale_index_coarse2fine = selected_scale_index,
            density_coarse2fine = self.selected_density_map,
            indices_shape_coarse2fine = self.latent_grains_shapes,
            indices_ratio_coarse2fine = selected_grain_ratio
        )
        
        self.selected_mask = selected_mask[::-1] # From fine to coarse

        return self.scale_indices, self.selected_mask, self.selected_density_map

    def select_elements(self, h: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Selects elements from multi-resolution latent variables based on given masks.

        Parameters:
        h (List[torch.Tensor]):     A list of latent variables at different resolutions.
                                    From fine to coarse
        masks (List[torch.Tensor]): Boolean masks indicating which elements to select.
                                    From fine to coarse
        
        Returns:
        selected_latent (torch.Tensor): Concatenated selected elements, shape (nbatch, nchannel * total_selected).
        """
        assert len(h) == len(masks), "The number of latent variables and masks must match."
        
        num_scales = len(h)
        selected_latents = []
        self.selected_locations = []  # Reset stored locations
        
        for i, (h_i, mask) in enumerate(zip(h, masks)):
            nbatch, nchannel, *_ = h_i.shape
            
            # Ensure mask is boolean and broadcast it
            mask = mask.bool().expand(nbatch, nchannel, *mask.shape[2:])  
            
            # Select unmasked elements
            selected_elements = h_i[mask].view(nbatch, nchannel, -1)  # (nbatch, nchannel, selected)
            selected_elements = selected_elements.reshape(nbatch, -1)  # (nbatch, nchannel * selected)
            
            selected_latents.append(selected_elements)
            
            # Store locations of selected elements
            locations = torch.nonzero(mask[0, 0], as_tuple=False)  # Store (x, y, z) positions
            self.selected_locations.append(locations)
            
            # print(f"Level {num_scales - i}: All--{h_i.shape}. Unmasked--{selected_elements.shape}. Location--{locations.shape}")

        # Concatenate all selected elements along the last dimension
        selected_latent = torch.cat(selected_latents, dim=-1)
        
        return selected_latent

    def restore_elements(self, selected_latent: torch.Tensor):
        """
        Restores the selected elements back to their original 5D shapes using stored locations.

        Parameters:
        selected_latent (torch.Tensor): Concatenated selected elements, shape (nbatch, nchannel * total_selected).

        Returns:
        restored_h (List[torch.Tensor]): List of restored latent variables with selected elements placed back.
        """
        h_shapes = self.all_latent_shapes
        assert len(h_shapes) == len(self.selected_locations), "Shape list must match stored locations."

        num_scales = len(h_shapes)
        restored_h = []
        start_idx = 0  # Index for selected_latent
        
        nbatch, nchannel_total = selected_latent.shape
        nchannel = h_shapes[0][1]
        selected_latent = selected_latent.view(nbatch, nchannel, -1)  # Reshape to (nbatch, nchannel, total_selected)

        for i, shape in enumerate(h_shapes):
            nbatch, nchannel, nx, ny, nz = shape

            # Initialize restore tensor with zeros (no gradients required)
            restore_tensor = torch.zeros((nbatch, nchannel, nx, ny, nz), 
                                         device=selected_latent.device, 
                                         dtype=selected_latent.dtype)

            # Get the number of selected elements for the current scale
            num_selected = self.selected_locations[i].shape[0]
            selected_chunk = selected_latent[:, :, start_idx:start_idx + num_selected]

            # Prepare indices for scatter operation
            idx_x, idx_y, idx_z = self.selected_locations[i][:, 0], self.selected_locations[i][:, 1], self.selected_locations[i][:, 2]

            # Expand indices to match batch and channel dimensions
            batch_indices = torch.arange(nbatch, device=selected_latent.device).view(-1, 1).expand(-1, num_selected)
            channel_indices = torch.arange(nchannel, device=selected_latent.device).view(1, -1, 1).expand(nbatch, -1, num_selected)

            # Use index_put to restore selected elements (preserves gradient tracking)
            restore_tensor = restore_tensor.index_put((batch_indices, channel_indices, idx_x, idx_y, idx_z), selected_chunk, accumulate=True)

            # Append to list
            restored_h.append(restore_tensor)
            start_idx += num_selected  # Update index
        
            # print(f"Level {num_scales - i} restored: {restore_tensor.shape} with {num_selected} elements.")
        
        return restored_h

    def upsampling(self, x, scale_factor, mode='nearest'):
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)

    def align_hidden_feature(self, h):
        h_grains = [h[0]] # finest scale
        for i, grain_factor in enumerate(self.all_grain_factors): # medium to coarsest scale
            h_grains.append(self.upsampling(h[i+1], grain_factor, mode=self.upsampling_mode))
        return h_grains

    def ensemble_to_latent(self, selected_latent):
        restored_h = self.restore_elements(selected_latent)
        h_grains = self.align_hidden_feature(restored_h)
        h_mixed = torch.sum(torch.stack(h_grains), dim=0)
        return h_mixed


