import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections.abc import Sequence
from typing import (Union, Optional, Dict, List, Tuple)
from os.path import join

from .modules import DownsampleEncoder, DownsampleEncoder2, \
    AdaptiveMaskModule, SimplifiedAdaptiveMaskModule,\
        SimpleDecoder, MultiScaleDownSample
from General import Encoder, Decoder, VariationalEncoder
from General import get_activation


class AdaptiveDecoder(nn.Module):
    def __init__(
            self,
            noutputs: int,
            spatial_dims: int,
            hidden_dims: List[int],
            latent_dims: int,
            proj_dims: Union[int, List[int]],
            blk_idx: Sequence[int], 
            decoder_scale_factors: Union[List[Sequence[int]], List[int]], 
            kernel_size: Union[Sequence[int], int],
            padding: Union[Sequence[int], int],
            strides: Union[Sequence[int], int] = 1,
            num_residual_layer: int = 5,
            adn_ordering: str = "NDA",
            act: Union[Tuple, str] = ("prelu", {"init": 0.2}),
            norm: Union[Tuple, str] = None,
            num_groups: int = 4,
            out_act:str = 'linear',
            **kwargs
            ):

        super().__init__()

        # Common configuration
        self.kwargs = kwargs
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.blk_idx = blk_idx
        self.proj_dims = proj_dims if proj_dims else hidden_dims[-1]
        self.decoder_scale_factors = decoder_scale_factors
        adn_ordering = "NDA" if adn_ordering is None else adn_ordering
        norm = ("group", {"num_groups": num_groups}) if norm is None else norm # remove if performance is degraded
        act = ("prelu", {"init": 0.2}) if act is None else act # ("prelu", {"init": 0.2}) # replace if performance is degraded
        self.adn_args = {"adn_ordering": adn_ordering, "act": act, "norm": norm}
        self.memory_efficient = kwargs.get('memory_efficient', True)

        # Special configuration
        use_default_adn_config = kwargs.get("use_default_adn_config", True)
        if use_default_adn_config: 
            num_groups, self.adn_args = None, None

        # Projector configuration
        self.num_proj_layers = kwargs.get('num_proj_layers', 1)
        self.out_proj_in_dim = self.proj_dims if isinstance(self.proj_dims, int) else self.proj_dims[-1]
        self.project_version = kwargs.get('project_version', 'old')

        # Decoder configuration
        self.use_residual_only = kwargs.get('use_residual_only', True)
        self.use_deconv_only = kwargs.get('use_deconv_only', True)
        self.index_sampling_type = kwargs.get('index_sampling_type', 'uniform')
        self.use_convtranspose = kwargs.get('use_convtranspose', True)
        self.use_different_kernel = kwargs.get('use_different_kernel', False)

        # Build layers
        # self.masker = AdaptiveMaskModule(scale_factors, blk_idx, **kwargs)

        self.decode = SimpleDecoder(
            spatial_dims, 
            scale_factors=self.decoder_scale_factors, 
            in_channels=self.latent_dims, 
            out_channels=self.proj_dims, 
            kernel_size=kernel_size, 
            padding=padding,
            strides=strides,
            adn_args=self.adn_args,
            num_groups=num_groups,
            num_residual_layer=num_residual_layer, 
            use_residual_only=self.use_residual_only, 
            use_deconv_only=self.use_deconv_only, 
            use_convtranspose=self.use_convtranspose,
            use_different_kernel=self.use_different_kernel,
            )
        
        self.out_proj = nn.Sequential(
            *[
                nn.Conv3d(in_channels=self.out_proj_in_dim, out_channels=noutputs, kernel_size=1, stride=1, padding=0), get_activation(out_act)
            ]
        ) 

    def forward(self, h_grains):
        """
        Forward pass with reshaping to restore (nbatch, num_items, ...).

        Args:
            h_grains (list of Tensors): A list of `num_items` tensors, each of shape `(nbatch, ...)`.

        Returns:
            list of Tensors: A list of `num_items` tensors, each of shape `(nbatch, ...)`.
        """

        # h_grains, indices, indices_repeat = self.masker(h, indices=indices, indices_repeat=indices_repeat)

        if self.memory_efficient:
            return self.decode_with_floop(h_grains)
        else:
            return self.decode_with_stack(h_grains)

    def decode_with_stack(self, h_grains):
        """
        Decodes a list of tensors efficiently using batch processing.

        Args:
            h_grains (list of Tensors): A list of K tensors, each of shape (B, C, X, Y, Z).

        Returns:
            list of Tensors: A list of K tensors, each of shape (B, ...).
        """

        # Concatenate along the batch dimension (B * K, C, X, Y, Z)
        h_grains_stacked = torch.cat(h_grains, dim=0)

        # Pass through the decoder in batch mode
        decoded = self.decode(h_grains_stacked)

        # Apply final projection
        recons_stacked = self.out_proj(decoded)

        # Split back into K tensors of shape (B, ...)
        return list(recons_stacked.chunk(len(h_grains), dim=0))
    
    def decode_with_floop(self, h_grains):
        """
        Decodes a list of tensors using a for-loop.

        Args:
            h_grains (list of Tensors): A list of K tensors, each of shape (B, C, X, Y, Z).

        Returns:
            list of Tensors: A list of K tensors, each of shape (B, ...).
        """
        return [self.out_proj(self.decode(x)) for x in h_grains]


class AdaptiveAutoEncoder(nn.Module): # Encoder + AdaptiveMaskModule + AdaptiveDecoder
    def __init__(self, units, model_config, **kwargs):
        super().__init__()

        ninputs, noutputs = units
        self.encoder = Encoder(ninputs, **model_config)
        self.masker = AdaptiveMaskModule(strides=model_config.get('scale_factors'), **model_config)
        self.decoder = AdaptiveDecoder(noutputs, decoder_scale_factors=self.masker.decode_factors, **model_config) 
        
    def forward(self, x, indices=None, indices_repeat=None):
        h = self.encoder(x)
        h_grains, indices, indices_repeat = self.masker(h, indices=indices, indices_repeat=indices_repeat)
        out = self.decoder(h_grains)
        return out, indices, indices_repeat


class AdaptiveAutoEncoder2(nn.Module): # DownsampleEncoder + AdaptiveMaskModule + AdaptiveDecoder
    def __init__(self, units, model_config, **kwargs):
        super().__init__()

        ninputs, noutputs = units
        self.encoder = DownsampleEncoder(ninputs, **model_config)
        self.masker = SimplifiedAdaptiveMaskModule(
            decode_factors=model_config.get('decode_factors'), 
            h_grain_factors=model_config.get('h_grain_factors'), 
            z_grain_factor=model_config.get('z_grain_factor'), 
            index_sampling_type=model_config.get('index_sampling_type', 'uniform'), 
            upsampling_mode=model_config.get('upsampling_mode', 'nearest')
            )
        self.decoder = AdaptiveDecoder(noutputs, decoder_scale_factors=self.masker.decode_factors, **model_config) 
        
    def forward(self, x, indices=None, indices_repeat=None):
        h = self.encoder(x)
        h_grains, indices, indices_repeat = self.masker(h, indices=indices, indices_repeat=indices_repeat)
        out = self.decoder(h_grains)
        return out, indices, indices_repeat


class AdaptiveAutoEncoder3(nn.Module): # DownsampleEncoder2 + AdaptiveMaskModule + AdaptiveDecoder
    def __init__(self, units, model_config, **kwargs):
        super().__init__()

        ninputs, noutputs = units
        self.encoder = DownsampleEncoder2(ninputs, **model_config)
        self.masker = SimplifiedAdaptiveMaskModule(
            decode_factors=model_config.get('decode_factors'), 
            h_grain_factors=model_config.get('h_grain_factors'), 
            z_grain_factor=model_config.get('z_grain_factor'), 
            index_sampling_type=model_config.get('index_sampling_type', 'uniform'), 
            upsampling_mode=model_config.get('upsampling_mode', 'nearest')
            )
        self.decoder = AdaptiveDecoder(noutputs, decoder_scale_factors=self.masker.decode_factors, **model_config) 
        
    def forward(self, x, indices=None, indices_repeat=None):
        h = self.encoder(x)
        h_grains, indices, indices_repeat = self.masker(h, indices=indices, indices_repeat=indices_repeat)
        out = self.decoder(h_grains)
        return out, indices, indices_repeat
