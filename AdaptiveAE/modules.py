import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence
from typing import (Union, Optional, Dict, List, Tuple)

from General.layers import Convolution, ResBlock, AttentionBlock, Downsample, Upsample
from General.lpips import LPIPS
from General.modules import Encoder, Decoder
from General.functions import calculate_grain_factors, compute_paddings_for_conv, compute_paddings_for_deconv


class CoarseningModule(nn.Module):
    def __init__(self, scale_factors):
        """
        Initialize the CoarseningModule.

        Args:
            scale_factors (list of tuple): Each element is a scaling factor (sx, sy, sz),
                                           representing the downsampling ratios along (H, W, D).
        """
        super().__init__()
        self.scale_factors = scale_factors

    def forward(self, x):
        """
        Perform downsampling on the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, nchannel, H, W, D).

        Returns:
            list of torch.Tensor: A list of downsampled tensors.
        """
        coarsened_tensors = []
        for scale_factor in self.scale_factors:
            # Compute the new size
            new_size = [
                x.size(2) // scale_factor[0],  # H
                x.size(3) // scale_factor[1],  # W
                x.size(4) // scale_factor[2],  # D
            ]
            # Downsample using interpolate
            coarsened = F.interpolate(
                x,
                size=new_size,
                mode="trilinear",
                align_corners=False
            )
            coarsened_tensors.append(coarsened)
        return coarsened_tensors


class SelectiveDownSampling(nn.Module):
    def __init__(self, scale_factors):
        """
        Downsampling module that performs downsampling while preserving distribution properties.

        Args:
            scale_factors (list of tuple): Each element is a scaling factor (sx, sy, sz),
                                           representing the downsampling ratios along (H, W, D).
        """
        super().__init__()
        self.scale_factors = scale_factors

    def forward(self, x: torch.Tensor):
        """
        Perform downsampling using stride-based selection (subsampling without interpolation).

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, H, W, D).

        Returns:
            list of torch.Tensor: A list of downsampled tensors.
        """
        coarsened_tensors = [x]
        for scale_factor in self.scale_factors:
            sx, sy, sz = scale_factor
            # Use fixed stride slicing to downsample
            coarsened = x[:, :, ::sx, ::sy, ::sz]  # Stride-based subsampling
            coarsened_tensors.append(coarsened)

        return coarsened_tensors


class PyramidDownSampling(nn.Module):
    def __init__(self, scale_factors, sigma=1.0):
        """
        Downsampling module using Gaussian Blur + Stride-Based Downsampling.

        Args:
            scale_factors (list of tuple): Each element is a scaling factor (sx, sy, sz),
                                           representing the downsampling ratios along (H, W, D).
            sigma (float): Standard deviation for Gaussian kernel.
        """
        super().__init__()
        self.scale_factors = scale_factors
        self.sigma = sigma if isinstance(sigma, list) else [sigma]*len(scale_factors)

        self.kernels = nn.ParameterDict()
        for idx, scale_factor in enumerate(self.scale_factors):
            kx, ky, kz = self.make_odd(scale_factor)  # Ensure odd kernel sizes
            self.kernels[str(idx)] = nn.Parameter(
                self.create_gaussian_kernel(kx, ky, kz, self.sigma[idx]), requires_grad=False)

    def make_odd(self, k):
        """Ensure kernel size is always odd (PyTorch `conv3d` requires odd kernel sizes)."""
        return tuple(k_i + 1 if k_i % 2 == 0 else k_i for k_i in k)

    def create_gaussian_kernel(self, kx, ky, kz, sigma):
        """
        Create a precomputed 3D Gaussian Kernel (H, W, D).

        Args:
            kx, ky, kz: Kernel sizes in height, width, and depth.

        Returns:
            torch.Tensor: Precomputed Gaussian Kernel of shape [1, 1, D, H, W].
        """
        def create_gaussian_1d(size, sigma):
            kernel = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
            kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
            kernel /= kernel.sum()
            return kernel.view(-1)
        
        kernel_H = create_gaussian_1d(kx, sigma)  # H
        kernel_W = create_gaussian_1d(ky, sigma)  # W
        kernel_D = create_gaussian_1d(kz, sigma)  # D

        # Convert to 3D kernel in user-defined order: (H, W, D)
        kernel = kernel_H.view(kx, 1, 1) * kernel_W.view(1, ky, 1) * kernel_D.view(1, 1, kz)
        
        # Adjust kernel dimensions to fit pytorch conv3d format: [channels, 1, D, H, W]
        kernel = kernel.permute(2, 0, 1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        return kernel
    
    def gaussian_blur(self, x: torch.Tensor, kernel_size, scale_idx):
        """
        Apply precomputed 3D Gaussian Blur.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, H, W, D).
            kernel_size (tuple): Kernel size (kx, ky, kz).

        Returns:
            torch.Tensor: Blurred tensor.
        """
        kernel = self.kernels[str(scale_idx)].to(x.device)
        padding = (kernel_size[2]//2, kernel_size[0]//2, kernel_size[1]//2)
        x = F.pad(x, 
          pad=(padding[2], padding[2],  # W padding
               padding[1], padding[1],  # H padding
               padding[0], padding[0]), # D padding
          mode='replicate')
        
        blurred_x = F.conv3d(x, kernel, padding=0, groups=x.shape[1])

        return blurred_x

    def forward(self, x: torch.Tensor, blur_only=False):
        """
        Perform downsampling using Gaussian Blur + Stride-Based Downsampling.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, H, W, D).

        Returns:
            list of torch.Tensor: A list of downsampled tensors forming a pyramid.
        """
        coarsened_tensors = [x]
        blurred_tensors = [x]

        for idx, scale in enumerate(self.scale_factors):
            sx, sy, sz = scale

            # Apply Gaussian blur
            blurred = self.gaussian_blur(x, 
                                         kernel_size=self.make_odd(scale),
                                         scale_idx=idx)
            # Stride-based downsampling
            downsampled = blurred[:, :, ::sx, ::sy, ::sz]

            coarsened_tensors.append(downsampled)
            blurred_tensors.append(blurred)

        if blur_only:
            return blurred_tensors
        
        return coarsened_tensors


class PyramidDownSampling2(nn.Module):
    def __init__(self, scale_factors, sigma=1.0, downsample_mode='stride'):
        """
        Downsampling module using Gaussian Blur + Stride-Based Downsampling.

        Args:
            scale_factors (list of tuple): Each element is a scaling factor (sx, sy, sz),
                                           representing the downsampling ratios along (H, W, D).
            sigma (float): Standard deviation for Gaussian kernel.
        """
        super().__init__()

        self.scale_factors = scale_factors
        self.sigma = sigma if isinstance(sigma, list) else [sigma]*len(scale_factors)
        self.downsample_mode = downsample_mode

        self.scale_intervals = self.convert_scales_to_kernel_sizes(scale_factors)
        self.kernel_sizes = [self.make_odd(k) for k in self.scale_intervals]  # Ensure odd kernel sizes
        self.kernels = nn.ParameterDict()
        for idx, kernel_size in enumerate(self.kernel_sizes):
            # kx, ky, kz = kernel_size
            self.kernels[str(idx)] = nn.Parameter(
                # self.create_gaussian_kernel2(kx, ky, kz, self.sigma[idx]), requires_grad=False
                self.create_gaussian_kernel(kernel_size, self.sigma[idx]), requires_grad=False
            )

    def make_odd(self, k):
        """Ensure kernel size is always odd (PyTorch `conv3d` requires odd kernel sizes)."""
        return tuple(k_i + 1 if k_i % 2 == 0 else k_i for k_i in k)
    
    def convert_scales_to_kernel_sizes(self, scale_factors):
        kernel_sizes = [scale_factors[0]]
        curr_scale = scale_factors[0]
        for scale_factor in scale_factors[1:]:
            new_size = [(a//b) for (a, b) in zip(scale_factor, curr_scale)]
            kernel_sizes.append(new_size)
            curr_scale = scale_factor
        return kernel_sizes
    
    def create_gaussian_kernel(self, kernel_size, sigma):
        coords = [torch.arange(k) - (k - 1) / 2 for k in kernel_size]
        grid_z, grid_y, grid_x = torch.meshgrid(*coords, indexing='ij')
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2 + grid_z ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.permute(2, 0, 1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel
        
    def create_gaussian_kernel2(self, kx, ky, kz, sigma):
        """
        Create a precomputed 3D Gaussian Kernel (H, W, D).

        Args:
            kx, ky, kz: Kernel sizes in height, width, and depth.

        Returns:
            torch.Tensor: Precomputed Gaussian Kernel of shape [1, 1, D, H, W].
        """
        def create_gaussian_1d(size, sigma):
            kernel = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
            kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
            kernel /= kernel.sum()
            return kernel.view(-1)
        
        kernel_H = create_gaussian_1d(kx, sigma)  # H
        kernel_W = create_gaussian_1d(ky, sigma)  # W
        kernel_D = create_gaussian_1d(kz, sigma)  # D

        # Convert to 3D kernel in user-defined order: (H, W, D)
        kernel = kernel_H.view(kx, 1, 1) * kernel_W.view(1, ky, 1) * kernel_D.view(1, 1, kz)
        
        # Adjust kernel dimensions to fit pytorch conv3d format: [channels, 1, D, H, W]
        kernel = kernel.permute(2, 0, 1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        return kernel
    
    def gaussian_blur(self, x: torch.Tensor, scale_idx):
        """
        Apply precomputed 3D Gaussian Blur.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, H, W, D).
            kernel_size (tuple): Kernel size (kx, ky, kz).

        Returns:
            torch.Tensor: Blurred tensor.
        """
        
        kernel_size = self.kernel_sizes[scale_idx]
        kernel = self.kernels[str(scale_idx)].to(x.device)
        padding = (kernel_size[2]//2, kernel_size[0]//2, kernel_size[1]//2)
        x = F.pad(x, 
          pad=(padding[2], padding[2],  # W padding
               padding[1], padding[1],  # H padding
               padding[0], padding[0]), # D padding
          mode='replicate')
        B, C, D, H, W = x.shape
        kernel = kernel.repeat(C, 1, 1, 1, 1)  # (C, 1, kD, kH, kW)

        blurred_x = F.conv3d(x, kernel, padding=0, groups=C)
        return blurred_x

    def forward(self, x: torch.Tensor, blur_only=False):
        """
        Perform downsampling using Gaussian Blur + Stride-Based Downsampling.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, H, W, D).

        Returns:
            list of torch.Tensor: A list of downsampled tensors forming a pyramid.
        """
        coarsened_tensors = [x]
        blurred_tensors = [x]

        for idx, scale in enumerate(self.scale_intervals):
            sx, sy, sz = scale

            # Apply Gaussian blur
            blurred = self.gaussian_blur(coarsened_tensors[-1], scale_idx=idx)

            if self.downsample_mode == 'stride':
                # Stride-based downsampling
                downsampled = blurred[:, :, ::sx, ::sy, ::sz]
            else:
                # Interpolation-based downsampling
                downsampled = F.interpolate(blurred, scale_factor=(1/sx, 1/sy, 1/sz), mode='trilinear', align_corners=False)
                        
            coarsened_tensors.append(downsampled)
            
            upsampled = F.interpolate(downsampled, scale_factor=self.scale_factors[idx], mode='trilinear', align_corners=False)
            blurred_tensors.append(upsampled)

        if blur_only:
            return blurred_tensors
        
        return coarsened_tensors


class AdaptiveMaskModule(nn.Module):
    def __init__(
            self, 
            strides=None,
            blk_idx=None,
            **kwargs
            ):
        super().__init__()
        
        """
        decode_factors:    scale factors for each layer in decoder to upsample the latent variable at finest scale
                           E.g., [(2, 2, 1), (2, 2, 1), ...]
        all_grain_factors: grain factors for stretching multiscale latent variables to the same shape,
                           excluding the finest scale. 
                           Start from the 2nd finest scale to coarsest scale
                           E.g., [(2, 2, 1), (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1)]
        """

        self.decode_factors, self.h_grain_factors, self.z_grain_factor = calculate_grain_factors(blk_idx, strides)
        self.all_grain_factors = self.h_grain_factors + [self.z_grain_factor] # 
        self.index_sampling_type = kwargs.get('index_sampling_type', 'uniform')
        self.upsampling_mode = kwargs.get('masker_upsample_mode', 'nearest')
        self.blk_idx = blk_idx
        self.strides = strides

    def upsampling(self, x, scale_factor, mode='nearest'):
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)

    def _repeat_interleave(self, x, repeats):
        return x.repeat_interleave(
            repeats[0], dim=2).repeat_interleave(
                repeats[1], dim=3).repeat_interleave(repeats[2], dim=4)

    def _generate_random_indices(self, N, z):
        indices_shape = list(z.size())
        indices_shape[1] = 1 # (N, 1, X, Y, Z) -- same random index over channel dimension
        indices = torch.randint(low=0, high=N, size=indices_shape, device=z.device)
        indices_repeat = self._repeat_interleave(indices, self.z_grain_factor)
        return indices, indices_repeat
    
    def _generate_random_indices_top2bottom(self, N, z):
        Nm1 = N - 1 # number of resolution scales
        current_scale = Nm1 - 1 # initialize the pointer of current scale
        indices_shape = list(z.size()) # (B, C, H, W, D)
        indices_shape[1] = 1 # (B, 1, H, W, D): all channels of a single pixel share the same index

        # initialize the index map from top to bottom:
        indices_repeat = self._repeat_interleave(Nm1*torch.ones(indices_shape, device=z.device), self.z_grain_factor) # top to bottom
        
        for grain_factor in self.all_grain_factors:
            tmp_indices_shape = list(indices_repeat.size()) # [1, 1, 64, 64, 1]
            tmp_indices_shape[2:] = [a//b for (a, b) in zip(tmp_indices_shape[2:], grain_factor)] 
            # [1, 1, 32, 32, 1], [1, 1, 16, 16, 1], [1, 1, 8, 8, 1], [1, 1, 4, 4, 1]
            tmp_indices = torch.randint(low=0, high=Nm1, size=tmp_indices_shape, device=z.device)
            tmp_indices_repeat = self._repeat_interleave(tmp_indices, grain_factor)
            indices_repeat[tmp_indices_repeat==0] = current_scale
            current_scale -= 1
        return indices_repeat, indices_repeat

    def _generate_random_indices_bottom2top(self, N, z):
        Nm1 = N - 1 # number of resolution scales
        threshold = 14

        # initialize the index map from bottom to top:
        current_scale = 0 # initialize the pointer of current scale
        indices_shape = list(z.size()) # (B, C, H, W, D)
        indices_shape[1] = 1 # (B, 1, H, W, D): all channels of a single pixel share the same index
        reverse_all_grain_factors = self.all_grain_factors[::-1] # (32, 32, 1), (16, 16, 1), (8, 8, 1), (4, 4, 1), (2, 2, 1)
        indices_repeat = self._repeat_interleave(Nm1*torch.ones(indices_shape, device=z.device), self.z_grain_factor) 
        tmp_indices_shape = indices_shape
        prev_factor = reverse_all_grain_factors[0]

        # iterate over different resolution scales:
        for grain_factor in reverse_all_grain_factors: 
            tmp_indices_shape[2:] = [x*(a//b) for (x, a, b) in zip(tmp_indices_shape[2:], prev_factor, grain_factor)] 
            tmp_indices = torch.randint(low=0, high=100, size=tmp_indices_shape, device=z.device)
            tmp_indices_repeat = self._repeat_interleave(tmp_indices, grain_factor)
            update_condition = (tmp_indices_repeat <= int(threshold)) & (indices_repeat > current_scale)
            indices_repeat[update_condition==True] = current_scale
            current_scale += 1
            threshold = threshold//0.73
            prev_factor = grain_factor
        return indices_repeat, indices_repeat

    def _align_hidden_feature(self, h):
        h_grains = [h[0]] # finest scale
        for i, grain_factor in enumerate(self.all_grain_factors): # medium to coarsest scale
            h_grains.append(self.upsampling(h[i+1], grain_factor, mode=self.upsampling_mode))
        return h_grains[::-1]

    def _get_random_indices(self, h, indices, indices_repeat):

        num_levels = len(h)

        if indices is None:
            if indices_repeat is None: # If both are None, generate new indices and their repeat
                index_generators = {
                    'uniform': self._generate_random_indices,
                    'bottom2top': self._generate_random_indices_bottom2top,
                    'top2bottom': self._generate_random_indices_top2bottom
                }

                if self.index_sampling_type not in index_generators:
                    raise ValueError("Only support 'uniform', 'bottom2top', and 'top2bottom'")
                
                indices, indices_repeat = index_generators[self.index_sampling_type](num_levels, h[-1])

        else:
            if indices_repeat is None: # If indices is not None, but indices_repeat is None, generate indices_repeat
                indices_repeat = self._repeat_interleave(indices, self.z_grain_factor)

        return indices, indices_repeat
    
    def forward(self, h, indices=None, indices_repeat=None):

        num_levels = len(h)

        # generate random indices (designed for controllable mask)
        indices, indices_repeat = self._get_random_indices(h, indices, indices_repeat)

        # align the spatial dimensions of hidden features
        h_grains = self._align_hidden_feature(h)

        # combine hidden features with different granularity together
        h_combined = torch.where(indices_repeat==0, h_grains[0], h_grains[1])
        for i in range(1, num_levels):
            h_combined = torch.where(indices_repeat==i, h_grains[i], h_combined)
        h_grains = [h_combined] + h_grains

        return h_grains, indices, indices_repeat


class SimplifiedAdaptiveMaskModule(AdaptiveMaskModule):
    def __init__(
            self, 
            decode_factors,
            h_grain_factors,
            z_grain_factor,
            index_sampling_type='uniform', 
            upsampling_mode='nearest'
        ):
        """
        Simplified AdaptiveMaskModule that takes `decode_factors`, `h_grain_factors`, and `z_grain_factor` directly,
        avoiding the need for `blk_idx` and `strides`.

        Args:
            decode_factors (List[Tuple[int, int, int]]): Scale factors for each layer in the decoder.
            h_grain_factors (List[Tuple[int, int, int]]): Grain factors for stretching multiscale latent variables.
            z_grain_factor (Tuple[int, int, int]): The coarsest grain factor.
        """
        # Call super() without blk_idx and strides
        super(AdaptiveMaskModule, self).__init__()   # super().__init__(**kwargs)

        # Directly use given factors instead of calculating them
        self.decode_factors = decode_factors
        self.h_grain_factors = h_grain_factors
        self.z_grain_factor = z_grain_factor
        self.all_grain_factors = self.h_grain_factors + [self.z_grain_factor]

        # Additional arguments
        self.index_sampling_type = index_sampling_type
        self.upsampling_mode = upsampling_mode


class MultiScaleDownSample(nn.Module):
    def __init__(self, 
                 **kwargs) -> None:

        super().__init__()

        self.latent_scale_factors = kwargs.get('latent_scale_factors', [])
        self.gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', 1.0)
        
        # Add the new PyramidDownSampling layer
        self.down = PyramidDownSampling(self.latent_scale_factors, 
                                        self.gaussian_blur_sigma)
        
    def forward(self, x):
        return self.down(x)


class DownsampleEncoder(Encoder):
    def __init__(
            self, 
            *args,  # Pass all other arguments to Encoder
            **kwargs
        ) -> None:
        """
        Forward pass of DownsampleEncoder.

        Args:
            latent_scale_factors (Union[List[Sequence[int]], List[int]]): Downsampling factors for the latent space.
            kl_mode (str): Method for computing KL divergence, either "sum" or "mean".
            *args: Arguments for the base Encoder.
            **kwargs: Keyword arguments for the base Encoder.
        """
        super().__init__(*args, **kwargs)  # Initialize Encoder with all its parameters

        self.latent_scale_factors = kwargs.get('latent_scale_factors', [])
        self.gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', 1.0)
        # Add the new PyramidDownSampling layer
        self.latent_down = PyramidDownSampling(self.latent_scale_factors, 
                                               self.gaussian_blur_sigma)
    
    def forward(self, x):
        """
        Variational Encoder with Downsampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after encoding and downsampling.
        """
        x = super().forward(x)[0]  # Use Encoder's forward pass to extract the highest level
        x = self.latent_down(x)
        return x


class DownsampleEncoder2(Encoder):
    def __init__(
            self, 
            *args,  # Pass all other arguments to Encoder
            **kwargs
        ) -> None:
        """
        Forward pass of DownsampleEncoder.

        Args:
            latent_scale_factors (Union[List[Sequence[int]], List[int]]): Downsampling factors for the latent space.
            kl_mode (str): Method for computing KL divergence, either "sum" or "mean".
            *args: Arguments for the base Encoder.
            **kwargs: Keyword arguments for the base Encoder.
        """
        super().__init__(*args, **kwargs)  # Initialize Encoder with all its parameters

        self.latent_scale_factors = kwargs.get('latent_scale_factors', [])
        self.gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', 1.0)
        self.latent_downsample_mode = kwargs.get('latent_downsample_mode', 'stride')

        # Add the new PyramidDownSampling layer
        self.latent_down = PyramidDownSampling2(self.latent_scale_factors, 
                                                sigma=self.gaussian_blur_sigma, 
                                                downsample_mode=self.latent_downsample_mode)
    
    def forward(self, x):
        """
        Variational Encoder with Downsampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after encoding and downsampling.
        """
        x = super().forward(x)[0]  # Use Encoder's forward pass to extract the highest level
        x = self.latent_down(x)
        return x


class DownsampleVariationalEncoder(Encoder):
    def __init__(
            self, 
            *args,  # Pass all other arguments to Encoder
            **kwargs
        ) -> None:
        """
        A modified Encoder with an additional SelectiveDownSampling layer.
        
        Args:
            latent_scale_factors (Union[List[Sequence[int]], List[int]]): Downsampling factors for the latent space.
            *args: Arguments for the base Encoder.
            **kwargs: Keyword arguments for the base Encoder.
        """
        super().__init__(*args, **kwargs)  # Initialize Encoder with all its parameters

        self.latent_scale_factors = kwargs.get('latent_scale_factors', [])
        self.kl_mode = kwargs.get('kl_mode', 'sum')

        # Add the new SelectiveDownSampling layer
        self.latent_down = SelectiveDownSampling(self.latent_scale_factors)
    
    def forward(self, x):
        """
        Forward pass of DownsampleVariationalEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (multi-res z, mu, logvar)
        """
        
        # Compute mu and logvar
        mu, logvar = torch.chunk(super().forward(x)[0], chunks=2, dim=1)

        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Multi-resolution selection via Downsampling
        multi_res_z = self.latent_down(z)

        return multi_res_z, mu, logvar  # Return all three for later KL divergence calculation

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent variable z.

        Args:
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log variance of latent distribution.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std  # Compute reparameterized z

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence loss.

        Args:
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log variance of latent distribution.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if self.kl_mode == "sum":
            return kld_loss.sum()  # Sum over all dimensions
        elif self.kl_mode == "mean":
            return kld_loss.mean()  # Mean over batch
        else:
            raise ValueError(f"Invalid kl_mode: {self.kl_mode}. Use 'sum' or 'mean'.")


class SimpleDecoder(nn.Module):
    def __init__(
            self, 
            spatial_dims: int, 
            scale_factors: List[int], 
            in_channels: int,
            out_channels: Union[int, List[int]], 
            kernel_size: Union[Sequence[int], int],
            padding: Union[Sequence[int], int],
            strides: Union[Sequence[int], int] = 1,
            adn_args: Optional[Dict] = None, 
            num_groups: Optional[int] = None, 
            num_residual_layer: Optional[int] = None,
            use_residual_only: bool = True,
            use_deconv_only: bool = False,
            use_convtranspose: bool = True,
            use_different_kernel: bool = False,
            **kwargs
            ):

        """
        Initializes the Decoder unit which is a sequence of deconvolution and residual blocks.

        Args:
            scale_factors (List[int]): Scaling factors for each deconvolution block.
            in_channels (int): Number of channels in the input tensor.
            out_channels (Union[int, List[int]]): Number of channels for the output tensor(s).
                                                  Can be a single integer or a list of integers.
            num_residual_layer (int): Number of residual blocks.
            mode (str): Mode of operation for deconvolution blocks.
            }
        """
        super().__init__()
        num_residual_layer = 5 if num_residual_layer is None else num_residual_layer
        num_groups = 4 if num_groups is None else num_groups
        if adn_args is None:
            norm = ("group", {"num_groups": num_groups})
            act = ("prelu", {"init": 0.2})
            adn_ordering = "NDA"
            adn_args = {"adn_ordering": adn_ordering, "act": act, "norm": norm}
        self.adn_args = adn_args
        self.use_residual_only = use_residual_only
        self.use_deconv_only = use_deconv_only
        self.use_convtranspose = use_convtranspose
        
        if isinstance(out_channels, int):
            out_channels_list = [out_channels] * len(scale_factors)  # Replicate for each deconv block
        elif isinstance(out_channels, list):
            assert len(out_channels) == len(scale_factors), "out_channels list must match the number of DeconvBlocks"
            out_channels_list = out_channels
        else:
            raise ValueError("out_channels must be either an integer or a list of integers")
        
        self.layers = nn.ModuleList()
        for idx, (scale, out_dim) in enumerate(zip(scale_factors, out_channels_list)):
            # Project layers
            self.layers.append(
                Convolution(spatial_dims=spatial_dims, 
                            in_channels=in_channels, 
                            out_channels=out_channels_list[0], 
                            kernel_size=1,
                            padding=0,
                            **self.adn_args
                            )
                )
            
            # Block: Residual or Deconv layers
            for stage in range(num_residual_layer):
                if self.use_residual_only:
                    self.layers.append(
                        ResBlock(spatial_dims=spatial_dims,
                                 in_channels=out_dim,  # Input channels for the ResBlock
                                 out_channels=out_dim,  # Output channels for the ResBlock
                                 norm_num_groups=num_groups,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding
                                 )
                        )
                elif self.use_deconv_only:
                    self.layers.append(
                        Convolution(spatial_dims=spatial_dims,
                                    in_channels=out_dim,
                                    out_channels=out_dim,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    **self.adn_args
                                    )
                        )
                else:
                    # Combine both if neither flag is set
                    self.layers.append(
                        ResBlock(spatial_dims=spatial_dims,
                                 in_channels=out_dim,  # Input channels for the ResBlock
                                 out_channels=out_dim,  # Output channels for the ResBlock
                                 norm_num_groups=num_groups,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding
                                 )
                        )
                    self.layers.append(
                        Convolution(spatial_dims=spatial_dims,
                                    in_channels=out_dim,
                                    out_channels=out_dim,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    strides=strides,
                                    **self.adn_args
                                    )
                        )
                
            # Upsampling layers
            if use_convtranspose and use_different_kernel:
                deconv_kernel_size, deconv_stride, deconv_padding, deconv_output_padding = self._get_convtranspose_setup(scale)
                self.layers.append(
                    Upsample(spatial_dims=spatial_dims, 
                             in_channels=out_dim,
                             use_convtranspose=use_convtranspose, 
                             strides=deconv_stride,
                             kernel_size=deconv_kernel_size,
                             padding=deconv_padding, 
                             output_padding=deconv_output_padding,
                             adn_args=self.adn_args,
                            )
                    )
            else:
                self.layers.append(
                    Upsample(spatial_dims=spatial_dims, 
                             in_channels=out_dim,
                             use_convtranspose=use_convtranspose, 
                             strides=scale,
                             kernel_size=kernel_size,
                             padding=padding, 
                             adn_args=self.adn_args
                            )
                    )
            in_channels = out_dim

    def _get_convtranspose_setup(self, scale_factor):
        # Adjust kernel size, stride, and padding based on scale_factor for transpose convolution
        deconv_kernel_size = [int(s + 2 * (s // 2)) for s in scale_factor]
        deconv_stride = scale_factor
        deconv_padding = [int(s // 2) for s in scale_factor]
        deconv_output_padding = [0 for s in scale_factor]
        return deconv_kernel_size, deconv_stride, deconv_padding, deconv_output_padding
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

