import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
from typing import List, Optional, Union, Callable, Tuple

from .functions import compute_paddings_for_conv, compute_paddings_for_deconv
from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


def get_activation(act_layer: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'elu': nn.ELU(),
        'prelu': nn.PReLU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU()
    }
    return activations.get(act_layer, nn.Identity())


class PixelDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(PixelDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x, input_mask=None):
        if not self.training or self.dropout_prob == 0:
            return x
        
        if input_mask is not None:
            # Use the provided mask if it is given
            mask = input_mask.float()
        else:
            # Generate a random dropout mask
            mask = (torch.rand(x.size(0), 1, *x.size()[2:], device=x.device) > self.dropout_prob).float()

        return x * mask


class ADN(nn.Sequential):
    """
    Constructs a sequential module of optional activation (A), dropout (D), and normalization (N) layers
    with an arbitrary order::

        -- (Norm) -- (Dropout) -- (Acti) --

    Args:
        ordering: a string representing the ordering of activation, dropout, and normalization. Defaults to "NDA".
        in_channels: `C` from an expected input of size (N, C, H[, W, D]).
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        norm_dim: determine the spatial dimensions of the normalization layer.
            defaults to `dropout_dim` if unspecified.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout.
            defaults to `norm_dim` if unspecified.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

    Examples::

        # activation, group norm, dropout
        >>> norm_params = ("GROUP", {"num_groups": 1, "affine": False})
        >>> ADN(norm=norm_params, in_channels=1, dropout_dim=1, dropout=0.8, ordering="AND")
        ADN(
            (A): ReLU()
            (N): GroupNorm(1, 1, eps=1e-05, affine=False)
            (D): Dropout(p=0.8, inplace=False)
        )

        # LeakyReLU, dropout
        >>> act_params = ("leakyrelu", {"negative_slope": 0.1, "inplace": True})
        >>> ADN(act=act_params, in_channels=1, dropout_dim=1, dropout=0.8, ordering="AD")
        ADN(
            (A): LeakyReLU(negative_slope=0.1, inplace=True)
            (D): Dropout(p=0.8, inplace=False)
        )

    See also:

        :py:class:`monai.networks.layers.Dropout`
        :py:class:`monai.networks.layers.Act`
        :py:class:`monai.networks.layers.Norm`
        :py:class:`monai.networks.layers.split_args`

    """

    def __init__(
        self,
        ordering: str = "NDA",
        in_channels: Optional[int] = None,
        act: Union[tuple, str, None] = "RELU",
        norm: Union[tuple, str, None] = None,
        norm_dim: Optional[int] = None,
        dropout: Union[tuple, str, float, None] = None,
        dropout_dim: Optional[int] = None,
    ) -> None:
        
        super().__init__()

        op_dict = {"A": None, "D": None, "N": None}
        # define the normalization type and the arguments to the constructor
        if norm is not None:
            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            op_dict["N"] = get_norm_layer(name=norm, spatial_dims=norm_dim or dropout_dim, channels=in_channels)

        # define the activation type and the arguments to the constructor
        if act is not None:
            op_dict["A"] = get_act_layer(act)

        if dropout is not None:
            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            op_dict["D"] = get_dropout_layer(name=dropout, dropout_dim=dropout_dim or norm_dim)

        for item in ordering.upper():
            if item not in op_dict:
                raise ValueError(f"ordering must be a string of {op_dict}, got {item} in it.")
            if op_dict[item] is not None:
                self.add_module(item, op_dict[item])


class MLP(nn.Module):
    """
    Flexible MLP.

    Args:
        input_dim: int, feature dim of input (last dim)
        output_dim: int, feature dim of output (last dim)
        hidden_dims: Optional[List[int]] - explicit hidden layer sizes
        num_layers: int - if hidden_dims is None, create num_layers of size hidden_dim
        hidden_dim: int - used when hidden_dims is None
        activation: Union[str, Tuple[str, dict], Callable, None] - activation for hidden layers
                    if tuple (name, kwargs) will call get_activation with name and apply kwargs if supported
        final_activation: same type as activation, optional applied after final linear
        use_batchnorm: bool - apply BatchNorm1d after each hidden Linear
        dropout_ratio: float - dropout applied after activation in each hidden layer (0 disables)
        bias: bool - whether to use bias in Linear layers
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        num_layers: int = 0,
        activation: Union[str, Tuple[str, dict], Callable, None] = "gelu",
        final_activation: Union[str, Tuple[str, dict], Callable, None] = None,
        use_batchnorm: bool = False,
        dropout_ratio: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        hidden_dim: int = 128
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * num_layers if num_layers > 0 else []

        layers: List[nn.Module] = []
        in_dim = input_dim

        def _build_act(act_spec):
            if act_spec is None:
                return nn.Identity()
            if isinstance(act_spec, tuple) and isinstance(act_spec[0], str):
                name, kwargs = act_spec[0], (act_spec[1] if len(act_spec) > 1 else {})
                # project-level get_activation returns instantiated module without kwargs,
                # if kwargs needed, simple mapping here for common params
                act = get_activation(name.lower())
                # try to set common kwargs if provided (e.g. negative_slope)
                try:
                    if kwargs:
                        # recreate typical activations with kwargs when possible
                        if name.lower() in ("leakyrelu",) and "negative_slope" in kwargs:
                            return nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=kwargs.get("inplace", False))
                    return act
                except Exception:
                    return act
            if isinstance(act_spec, str):
                return get_activation(act_spec.lower())
            if callable(act_spec):
                return act_spec() if isinstance(act_spec, type) else act_spec
            return nn.Identity()

        hidden_act = _build_act(activation)
        final_act = _build_act(final_activation)

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=bias))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(hidden_act if isinstance(hidden_act, nn.Module) else hidden_act)
            if dropout_ratio and dropout_ratio > 0.0:
                layers.append(nn.Dropout(dropout_ratio))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias))
        if final_activation is not None:
            layers.append(final_act)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect last dim == input_dim, preserve leading dims
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = self.net(x_flat)
        out = y_flat.view(*orig_shape, y_flat.shape[-1])
        return out


class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
          (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Union[tuple, str, None] = "PRELU",
        norm: Union[tuple, str, None] = "INSTANCE",
        dropout: Union[tuple, str, float, None] = None,
        dropout_dim: Union[int, None] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Union[Sequence[int], int, None] = None,
        output_padding: Union[Sequence[int], int, None] = None,
    ) -> None:
        
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                padding, output_padding = compute_paddings_for_deconv(strides, kernel_size)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module("adn", ADN(
            ordering=adn_ordering,
            in_channels=out_channels,
            act=act,
            norm=norm,
            norm_dim=self.spatial_dims,
            dropout=dropout,
            dropout_dim=dropout_dim,
            ),
        )


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(self, 
                 spatial_dims: int, 
                 in_channels: int, 
                 use_convtranspose: bool = True, 
                 strides: Union[Sequence[int], int] = 2,
                 kernel_size: Union[Sequence[int], int] = 3,
                 padding: Union[Sequence[int], int] = 1,
                 conv_only: bool = False,
                 adn_args: dict = None,
                 output_padding: Union[Sequence[int], int, None] = None,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.strides = strides
        # Ensure adn_args is a dictionary
        if adn_args is None:
            adn_args = {}
            conv_only = True
        else:
            conv_only = False

        self.mode = kwargs.get("mode", "nearest")

        if use_convtranspose:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=strides,
                kernel_size=kernel_size,
                padding=padding,
                conv_only=conv_only,
                is_transposed=True, # norm=norm,
                output_padding=output_padding,
                **adn_args
            )
        else:
            # For non-transposed convolution, strides are typically set to 1
            # and kernel_size/padding might have default or specified values
            conv_kernel_size = kwargs.get("conv_kernel_size", kernel_size)
            conv_padding = kwargs.get("conv_padding", padding)
            # conv_kernel_size = kernel_size if kwargs.get("conv_kernel_size") is None else kwargs.get("conv_kernel_size", 3)
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=conv_kernel_size,
                padding=conv_padding,
                conv_only=conv_only,
                **adn_args, 
            )
        self.use_convtranspose = use_convtranspose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            return self.conv(x)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=self.strides, mode=self.mode)

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        adn_args: configurations for the ADN sequence such as:
            - "ordering": Order of activation, dropout, normalization (e.g., "ADN").
            - "act": Activation function and its parameters (e.g., "relu" or ("prelu", {"init": 0.2})).
            - "norm": Normalization type and its parameters (e.g., "batch" or ("instance", {"affine": True})).
            - "dropout": Dropout ratio (e.g., 0.1).
            - "dropout_dim": Spatial dimensions of dropout, applicable for 2D/3D convolutions (e.g., 1 for standard dropout).
        """

    def __init__(self, 
                 spatial_dims: int, 
                 in_channels: int,
                 strides: Union[Sequence[int], int] = 2,
                 kernel_size: Union[Sequence[int], int] = 3,
                 padding: Union[Sequence[int], int] = 1,
                 conv_only: bool = False, # norm: Union[Tuple, str] = 'group',
                 adn_args: dict = None,
                 **kwargs
                 ) -> None:
        super().__init__()

        if adn_args is None:
            adn_args = {} # Ensure adn_args is a dictionary
            conv_only = True
        else:
            conv_only = False

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=kernel_size,
            padding=padding,
            conv_only=conv_only,
            **adn_args
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        norm_num_groups: int, 
        norm_eps: float=1e-05, 
        kernel_size: Union[Sequence[int], int] = 3,
        padding: Union[Sequence[int], int] = 1,
        strides: Union[Sequence[int], int] = 1,
        out_channels: Optional[int] = None,
        conv_only: bool = True,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            conv_only=conv_only,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            conv_only=conv_only,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        # use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        # self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    # def _memory_efficient_attention_xformers(
    #     self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    # ) -> torch.Tensor:
    #     query = query.contiguous()
    #     key = key.contiguous()
    #     value = value.contiguous()
    #     x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
    #     return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # if self.use_flash_attention:
        #     x = self._memory_efficient_attention_xformers(query, key, value)
        # else:
        #     x = self._attention(query, key, value)
        x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual


class LocationEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, grid_shape):
        """
        Args:
            embedding_dim (int): The dimension of the location embedding.
            grid_shape (tuple): The shape of the 3D grid (D, H, W).
        """
        super().__init__()
        D, H, W = grid_shape

        # Generate location embeddings for each spatial dimension
        self.depth_embedding = torch.nn.Embedding(D, embedding_dim)
        self.height_embedding = torch.nn.Embedding(H, embedding_dim)
        self.width_embedding = torch.nn.Embedding(W, embedding_dim)

    def forward(self, x):
        """
        Add location embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Tensor with added location embeddings.
        """
        B, C, D, H, W = x.shape

        # Generate a grid of indices for each spatial dimension
        depth_idx = torch.arange(D, device=x.device).view(1, D, 1, 1).expand(B, D, H, W)
        height_idx = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(B, D, H, W)
        width_idx = torch.arange(W, device=x.device).view(1, 1, 1, W).expand(B, D, H, W)

        # Compute embeddings for each spatial dimension
        depth_emb = self.depth_embedding(depth_idx).permute(0, 4, 1, 2, 3)  # (B, Embedding_dim, D, H, W)
        height_emb = self.height_embedding(height_idx).permute(0, 4, 1, 2, 3)
        width_emb = self.width_embedding(width_idx).permute(0, 4, 1, 2, 3)

        # Combine embeddings
        location_emb = depth_emb + height_emb + width_emb

        # Add location embeddings to input tensor
        x = x + location_emb
        return x
