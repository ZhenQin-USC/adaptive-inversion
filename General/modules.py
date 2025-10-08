import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence
from typing import (Union, Optional, Dict, List, Tuple)

from .layers import MLP, Convolution, ResBlock, AttentionBlock, Downsample, Upsample, LocationEmbedding
from .lpips import LPIPS
from .functions import calculate_grain_factors, compute_paddings_for_conv, compute_paddings_for_deconv


class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
    Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
    that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
    58d9a2746493717a7c9252938da7efa6006f3739.

    This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
    to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
    on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

    Args:
        spatial_dims :  number of spatial spatial_dims.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
        decay: EMA decay. Defaults to 0.99.
        epsilon: epsilon value. Defaults to 1e-5.
        embedding_init: initialization method for the codebook. Defaults to "normal".
        ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings

        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )

        self.embedding: nn.Embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        self.embedding.weight.requires_grad = False

        self.commitment_cost: float = commitment_cost

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

        self.decay: float = decay
        self.epsilon: float = epsilon

        self.ddp_sync: bool = ddp_sync

        # Precalculating required permutation shapes
        self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
            range(1, self.spatial_dims + 1)
        )

    # @torch.cuda.amp.autocast(enabled=False)
    @torch.amp.autocast('cuda', enabled=False)
    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs: Encoding space tensors

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
            torch.Tensor: Quantization indices of shape [B,D,H,W,1]

        """
        encoding_indices_view = list(inputs.shape)
        del encoding_indices_view[1]

        inputs = inputs.float()

        # Converting to channel last format
        flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

        # Calculate Euclidean distances
        distances = (
            (flat_input**2).sum(dim=1, keepdim=True)
            + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
            - 2 * torch.mm(flat_input, self.embedding.weight.t())
        )

        # Mapping distances to indexes
        encoding_indices = torch.max(-distances, dim=1)[1]
        encodings = nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize and reshape
        encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices

    # @torch.cuda.amp.autocast(enabled=False)
    @torch.amp.autocast('cuda', enabled=False)
    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices: Tensor in channel last format which holds indices referencing atomic
                elements from self.embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

    @torch.jit.unused
    def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
        """
        TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
        example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

        Args:
            encodings_sum: The summation of one hot representation of what encoding was used for each
                position.
            dw: The multiplication of the one hot representation of what encoding was used for each
                position with the flattened input.

        Returns:
            None
        """
        if self.ddp_sync and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
        else:
            pass

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))

        # Encoding Loss
        loss = self.commitment_cost * nn.functional.mse_loss(quantized.detach(), inputs)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class VectorQuantizer(nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation. Defaults to None
    """

    def __init__(self, quantizer: nn.Module = None):
        super().__init__()

        self.quantizer: nn.Module = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices = self.quantizer(inputs)

        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices = self.quantizer(encodings)

        return encoding_indices


class PerceptualLoss(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        cache_dir: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if (spatial_dims == 2 or is_fake_3d) and "medicalnet_" in network_type:
            raise ValueError(
                "MedicalNet networks are only compatible with ``spatial_dims=3``."
                "Argument is_fake_3d must be set to False."
            )

        if cache_dir:
            torch.hub.set_dir(cache_dir)

        self.spatial_dims = spatial_dims
        self.perceptual_function = LPIPS(pretrained=pretrained, net=network_type, verbose=False)
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio

    def _calculate_axis_loss(self, input: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            input: input 5D tensor. BNHWD
            target: target 5D tensor. BNHWD
            spatial_axis: spatial axis to obtain the 2D slices.
        """

        def batchify_axis(x: torch.Tensor, fake_3d_perm: tuple) -> torch.Tensor:
            """
            Transform slices from one spatial axis into different instances in the batch.
            """
            slices = x.float().permute((0,) + fake_3d_perm).contiguous()
            slices = slices.view(-1, x.shape[fake_3d_perm[1]], x.shape[fake_3d_perm[2]], x.shape[fake_3d_perm[3]])

            return slices

        preserved_axes = [2, 3, 4]
        preserved_axes.remove(spatial_axis)

        channel_axis = 1
        input_slices = batchify_axis(x=input, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(x=target, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.spatial_dims == 3 and self.is_fake_3d:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(input, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(input, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(input, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        else:
            # 2D and real 3D cases
            loss = self.perceptual_function(input, target)

        return torch.mean(loss)


class Encoder(nn.Module):
    def __init__(
            self, 
            ninputs: int, 
            spatial_dims: int, 
            hidden_dims: List[int],
            latent_dims: int,
            blk_idx: Sequence[int], 
            attention_levels: Sequence[bool], 
            scale_factors: Union[List[Sequence[int]], List[int]], 
            kernel_size: Union[Sequence[int], int],
            padding: Union[Sequence[int], int],
            num_residual_layer: int = 3,
            adn_ordering: str = None,
            act: Union[Tuple, str] = None,
            norm: Union[Tuple, str] = None,
            num_groups: int = 4,
            use_hidden_proj: bool = False,
            attn_num_heads: int = 4,
            **kwargs
            ) -> None:

        super().__init__()

        assert len(scale_factors) == len(hidden_dims)-1, ValueError(
            f"The lengths of scale_factors ({len(scale_factors)}) and hidden_dims ({len(hidden_dims)-1}) are not consistent.")
        
        self.spatial_dims = spatial_dims
        self.ninputs = ninputs
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.scale_factors = scale_factors
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.padding = padding
        self.attention_levels = attention_levels
        self.blk_idx = blk_idx
        self.use_hidden_proj = use_hidden_proj
        self.norm = norm
        self.num_residual_layer = num_residual_layer
        self.attn_num_heads = attn_num_heads

        adn_ordering = "NDA" if adn_ordering is None else adn_ordering
        norm = ("group", {"num_groups": num_groups}) if norm is None else norm
        act = ("gelu", {}) if act is None else act # ("prelu", {"init": 0.2})
        self.adn_args = {"adn_ordering": adn_ordering, "act": act, "norm": norm}
        self.scale_kernel_size = kwargs.get('scale_kernel_size', [kernel_size]*len(scale_factors))
        
        # Input project
        self.in_proj = Convolution(spatial_dims=spatial_dims, 
                                   in_channels=ninputs, 
                                   out_channels=hidden_dims[0], 
                                   kernel_size=1, 
                                   padding=0, 
                                   **self.adn_args)
        
        # Block
        blocks = []
        for ind in range(len(hidden_dims)-1):
            blocks.append(nn.Sequential(*self._get_sub_block(ind)))
        self.blocks = nn.Sequential(*blocks)

        # Output project
        self.out_proj = Convolution(spatial_dims=spatial_dims, 
                                    in_channels=hidden_dims[-1], 
                                    out_channels=latent_dims, 
                                    conv_only=True, 
                                    kernel_size=1, 
                                    padding=0)

        # Hidden project
        if self.use_hidden_proj:
            self.hidden_proj = nn.Sequential(*[
                Convolution(spatial_dims=spatial_dims, 
                            in_channels=hidden_dims[idx+1], 
                            out_channels=latent_dims, 
                            conv_only=True,
                            kernel_size=1, 
                            padding=0
                            ) for idx in self.blk_idx])
        else:
            self.hidden_proj = nn.Sequential(*[nn.Identity() for idx in self.blk_idx])

    def _get_sub_block(self, ind):

        strides = self.scale_factors[ind]
        kernel_size = self.scale_kernel_size[ind]
        padding = compute_paddings_for_conv(strides, kernel_size)

        sub_block = []
        sub_block.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.hidden_dims[ind],
                out_channels=self.hidden_dims[ind+1], 
                # norm_num_groups=num_groups, # norm_eps=norm_eps,
                kernel_size=self.kernel_size, 
                padding=self.padding, 
                **self.adn_args
            )
        )
        
        # Add residual blocks
        sub_block.extend(
            [
                ResBlock(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.hidden_dims[ind+1],
                    out_channels=self.hidden_dims[ind+1],
                    norm_num_groups=self.num_groups, # norm_eps=norm_eps,
                    kernel_size=self.kernel_size, 
                    padding=self.padding) for _ in range(self.num_residual_layer)
                ]
            ) 

        # If attention is required at this level
        if self.attention_levels[ind]:
            num_heads = self.attn_num_heads if self.hidden_dims[ind+1] % self.attn_num_heads == 0 else 1
            sub_block.append(
                AttentionBlock(
                            spatial_dims=self.spatial_dims,
                            num_channels=self.hidden_dims[ind+1],
                            num_head_channels=self.hidden_dims[ind+1] // num_heads, 
                            norm_num_groups=self.num_groups,
                            )
            )

        # Downsample the feature maps 
        sub_block.append(
            Downsample(
                spatial_dims=self.spatial_dims, 
                in_channels=self.hidden_dims[ind+1], 
                strides=strides, 
                kernel_size=kernel_size, 
                padding=padding, 
                adn_args=self.adn_args,
            )
        )
        return sub_block
    
    def forward(self, x):
        intermediates = []

        x = self.in_proj(x)

        for ind, block in enumerate(self.blocks):
            x = block(x)

            if ind in self.blk_idx:
                cur_idx = len(intermediates)
                intermediates.append(self.hidden_proj[cur_idx](x))

        intermediates.append(self.out_proj(x))

        return intermediates


class Decoder(nn.Module):
    def __init__(
            self, 
            noutputs: int, 
            spatial_dims: int, 
            hidden_dims: List[int],
            latent_dims: int,
            blk_idx: Sequence[int], 
            attention_levels: Sequence[bool], 
            kernel_size: Union[Sequence[int], int],
            padding: Union[Sequence[int], int],
            scale_factors: Union[List[Sequence[int]], List[int]], 
            num_residual_layer: int = 3,
            adn_ordering: str = None,
            act: Union[Tuple, str] = None,
            norm: Union[Tuple, str] = None,
            num_groups: int = 4,
            use_hidden_proj: bool = False,
            attn_num_heads: int = 4,
            out_act: Optional[str] = None,
            **kwargs
            ) -> None:

        super().__init__()

        assert len(scale_factors) == len(hidden_dims)-1, ValueError(
            f"The lengths of scale_factors ({len(scale_factors)}) and hidden_dims ({len(hidden_dims)-1}) are not consistent.")

        self.spatial_dims = spatial_dims
        self.noutputs = noutputs
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.scale_factors = scale_factors
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.padding = padding
        self.attention_levels = attention_levels
        self.blk_idx = blk_idx
        self.use_hidden_proj = use_hidden_proj
        self.norm = norm
        self.num_residual_layer = num_residual_layer
        self.attn_num_heads = attn_num_heads
        self.reverse_blk_idx = [len(hidden_dims) - _ - 2 for _ in blk_idx]
        adn_ordering = "NDA" if adn_ordering is None else adn_ordering
        norm = ("group", {"num_groups": num_groups}) if norm is None else norm
        act = ("gelu", {}) if act is None else act # ("prelu", {"init": 0.2})
        self.adn_args = {"adn_ordering": adn_ordering, "act": act, "norm": norm}
        self.scale_kernel_size = kwargs.get('scale_kernel_size', [kernel_size]*len(scale_factors))
        self.use_convtranspose = kwargs.get('use_convtranspose', True)

        # Output project
        self.out_proj = Convolution(spatial_dims=spatial_dims, 
                                    in_channels=hidden_dims[0], 
                                    out_channels=noutputs, 
                                    kernel_size=1, 
                                    padding=0, 
                                    norm=None,
                                    act=out_act)
        
        # Block
        blocks = []
        for ind in range(len(hidden_dims)-1):
            blocks.append(nn.Sequential(*self._get_sub_block(ind)))
        blocks.reverse()
        self.blocks = nn.Sequential(*blocks)

        # Input project
        self.in_proj = Convolution(spatial_dims=spatial_dims, 
                                   in_channels=latent_dims, 
                                   out_channels=hidden_dims[-1], 
                                   conv_only=True, 
                                   kernel_size=1, 
                                   padding=0)

        # Hidden project
        if self.use_hidden_proj:
            self.hidden_proj = nn.Sequential(*[
                Convolution(spatial_dims=spatial_dims, 
                            in_channels=latent_dims, 
                            out_channels=hidden_dims[idx+1], 
                            conv_only=True,
                            kernel_size=1, 
                            padding=0
                            ) for idx in blk_idx[::-1]])
        else:
            self.hidden_proj = nn.Sequential(*[nn.Identity() for idx in self.blk_idx])

    def _get_sub_block(self, ind):

        strides = self.scale_factors[ind]
        if strides == 1 or strides == (1, 1, 1): # no strides -> default kernel size and padding
            kernel_size = self.kernel_size
            padding = self.padding
            output_padding = None
        else: # strides != 1 -> special kernel size and padding for deconv
            kernel_size = self.scale_kernel_size[ind]
            padding, output_padding = compute_paddings_for_deconv(strides, kernel_size)
        
        # Handle skip connections and convolution adjustments
        in_channels = self.hidden_dims[ind + 1] * 2 if ind in self.blk_idx else self.hidden_dims[ind + 1]

        sub_block = []
        sub_block.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=self.hidden_dims[ind + 1],
                kernel_size=self.kernel_size, 
                padding=self.padding, 
                **self.adn_args
            )
        )

        # Add residual blocks
        sub_block.extend(
            [
                ResBlock(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.hidden_dims[ind + 1],
                    out_channels=self.hidden_dims[ind + 1],
                    norm_num_groups=self.num_groups,
                    kernel_size=self.kernel_size, 
                    padding=self.padding) for _ in range(self.num_residual_layer)
                ]
            )


        # If attention is required at this level
        if self.attention_levels[ind]:
            num_heads = self.attn_num_heads if self.hidden_dims[ind + 1] % self.attn_num_heads == 0 else 1
            sub_block.append(
                AttentionBlock(
                    spatial_dims=self.spatial_dims,
                    num_channels=self.hidden_dims[ind + 1],
                    num_head_channels=self.hidden_dims[ind + 1] // num_heads, 
                    norm_num_groups=self.num_groups,
                )
            )

        # Upsample the feature maps 
        sub_block.append(
            Upsample(
                spatial_dims=self.spatial_dims, 
                use_convtranspose=self.use_convtranspose,
                in_channels=self.hidden_dims[ind + 1], 
                strides=strides, 
                kernel_size=kernel_size, 
                padding=padding, 
                output_padding=output_padding,
                adn_args=self.adn_args,
                conv_kernel_size=self.kernel_size, 
                conv_padding=self.padding
            )
        )
        
        # Reduce the channels
        sub_block.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.hidden_dims[ind + 1],
                out_channels=self.hidden_dims[ind],
                kernel_size=self.kernel_size, 
                padding=self.padding, 
                **self.adn_args
            )
        )

        return sub_block

    def forward(self, intermediates):
        encoder_intermediates = intermediates[::-1]
        z = encoder_intermediates.pop(0)
        z = self.in_proj(z)
        total_skipped = len(encoder_intermediates)
        for ind, block in enumerate(self.blocks):

            if ind in self.reverse_blk_idx:
                cur_idx = total_skipped - len(encoder_intermediates)
                encoder_output = encoder_intermediates.pop(0)
                z = torch.cat([z, self.hidden_proj[cur_idx](encoder_output)], dim=1)
            z = block(z)

        return self.out_proj(z)
    

class VariationalEncoder(Encoder):
    def __init__(self, *args, **kwargs) -> None:
        super(VariationalEncoder, self).__init__(*args, **kwargs)
        self.kl_mode = kwargs.get('kl_mode', 'mean') # mean or sum

        # Update the output projection to produce mu and logvar
        self.out_proj = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.hidden_dims[-1],
            out_channels=2*self.latent_dims,  # Double for mu and logvar
            conv_only=True,
            kernel_size=1,
            padding=0
        )

        # Update hidden projection if using hidden projections
        self.hidden_proj = nn.ModuleList([
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.hidden_dims[idx+1],
                out_channels=2*self.latent_dims,  # Double for mu and logvar
                conv_only=True,
                kernel_size=1,
                padding=0
            ) for idx in self.blk_idx])

    def forward(self, x):
        mu_intermediates = []
        logvar_intermediates = []

        x = self.in_proj(x)

        for ind, block in enumerate(self.blocks):
            x = block(x)
            if self.blk_idx is not None and ind in self.blk_idx:
                cur_idx = self.blk_idx.index(ind)
                hidden_out = self.hidden_proj[cur_idx](x)
                mu, logvar = torch.split(hidden_out, self.latent_dims, dim=1)
                mu_intermediates.append(mu)
                logvar_intermediates.append(logvar)

        # Compute mean and log variance for the final layer
        out = self.out_proj(x)
        mu, logvar = torch.split(out, self.latent_dims, dim=1)

        mu_intermediates.append(mu)
        logvar_intermediates.append(logvar)

        # Perform reparameterization trick
        intermediates = self.reparameterize(mu_intermediates, logvar_intermediates)

        # Calculate KL Divergence
        kld_list = self.kl_divergence(mu_intermediates, logvar_intermediates)

        return intermediates, mu_intermediates, logvar_intermediates, kld_list

    def reparameterize(self, mu_list, logvar_list):
        z_list = []
        for mu, logvar in zip(mu_list, logvar_list):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z_list.append(z)
        return z_list

    def kl_divergence(self, mu_list, logvar_list):
        kld_list = []
        for mu, logvar in zip(mu_list, logvar_list):
            # Compute the KL divergence term
            if self.kl_mode == "sum":
                # sum over all dims except batch
                reduce_dims = tuple(range(1, mu.dim()))
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=reduce_dims)
            else: # self.kl_mode == "mean":
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kld_list.append(kld.mean())
        return kld_list


class UNet(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            nfeatures: Union[int, List[int]],
            hidden_features: List[int],
            hidden_blk_idx: Sequence[int],
            hidden_attention_levels: Sequence[bool],
            scale_factors: Union[List[Sequence[int]], List[int]],
            kernel_size: Union[Sequence[int], int],
            padding: Union[Sequence[int], int],
            num_residual_layer: int = 3,
            adn_ordering: str = "NDA",
            act: Union[Tuple, str] = "gelu",
            norm: Union[Tuple, str] = None,
            num_groups: int = 4,
            use_hidden_proj: bool = False,
            attn_num_heads: int = 4,
            out_act: Optional[str] = None,
            **kwargs
    ) -> None:
        super(UNet, self).__init__()
        if isinstance(nfeatures, int):
            ninputs, noutputs = nfeatures, nfeatures
        elif isinstance(nfeatures, list) and len(nfeatures) == 2:
            ninputs, noutputs = nfeatures
        else:
            raise ValueError("nfeatures must be an int or a list of two integers.")

        self.use_embedding = False
        
        if kwargs.get('embedding_dim') and kwargs.get('grid_shape'):
            self.use_embedding = True
            self.embedding = LocationEmbedding(kwargs.get('embedding_dim'), kwargs.get('grid_shape'))

        self.encoder = Encoder(
            ninputs=ninputs,
            spatial_dims=spatial_dims,
            hidden_dims=hidden_features,
            latent_dims=hidden_features[-1],
            blk_idx=hidden_blk_idx,
            attention_levels=hidden_attention_levels,
            scale_factors=scale_factors,
            kernel_size=kernel_size,
            padding=padding,
            num_residual_layer=num_residual_layer,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            num_groups=num_groups,
            use_hidden_proj=use_hidden_proj,
            attn_num_heads=attn_num_heads,
        )

        self.decoder = Decoder(
            noutputs=noutputs,
            spatial_dims=spatial_dims,
            hidden_dims=hidden_features,
            latent_dims=hidden_features[-1],
            blk_idx=hidden_blk_idx,
            attention_levels=hidden_attention_levels,
            scale_factors=scale_factors,
            kernel_size=kernel_size,
            padding=padding,
            num_residual_layer=num_residual_layer,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            num_groups=num_groups,
            use_hidden_proj=use_hidden_proj,
            attn_num_heads=attn_num_heads,
            out_act=out_act,
        )

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        intermediates = self.encoder(x)
        out = self.decoder(intermediates)
        return out


