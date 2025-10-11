import torch
import torch.nn as nn
import kornia 
import torch.nn.functional as F
from typing import (Union, Optional, Dict, List, Tuple)
from einops import rearrange

from .get_kernels_3d import get_kernels_3d
from .lpips import LPIPS
from .registry import register_multifield_loss, register_singlefield_loss


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

    def _calculate_axis_loss(self, pred: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            pred: pred 5D tensor. BNHWD
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
        input_slices = batchify_axis(x=pred, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(x=target, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")

        if self.spatial_dims == 3 and self.is_fake_3d:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(pred, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(pred, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(pred, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        else:
            # 2D and real 3D cases
            loss = self.perceptual_function(pred, target)

        return torch.mean(loss)


class PixelWiseLoss3D(nn.Module):
    def __init__(self, loss_type='l1', p=2.0, reduce_dims=None):
        """
        Pixel-wise loss for 3D data.

        Args:
            loss_type: one of ['l1', 'mse', 'rel_l1', 'rel_mse', 'lp', 'rel_lp']
            p: float, p-norm (only used for 'lp' and 'rel_lp')
            reduce_dims: dims to reduce over (default (1,2,3,4) = C,D,H,W)
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.p = p
        self.reduce_dims = reduce_dims if reduce_dims is not None else (1, 2, 3, 4)

        assert self.loss_type in ['l1', 'mse', 'rel_l1', 'rel_mse', 'lp', 'rel_lp']

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, D, H, W)
        Returns scalar loss
        """
        assert pred.shape == target.shape, "Shape mismatch"

        if self.loss_type == 'l1':
            return F.l1_loss(pred, target)

        elif self.loss_type == 'mse':
            return F.mse_loss(pred, target)

        elif self.loss_type == 'lp':
            diff = torch.abs(pred - target) ** self.p
            return diff.mean()

        elif self.loss_type == 'rel_l1':
            num = torch.abs(pred - target).mean(dim=self.reduce_dims)
            den = torch.abs(target).mean(dim=self.reduce_dims) + 1e-8
            rel = num / den
            return rel if rel.ndim == 0 else rel.mean()

        elif self.loss_type == 'rel_mse':
            num = ((pred - target) ** 2).mean(dim=self.reduce_dims)
            den = (target ** 2).mean(dim=self.reduce_dims) + 1e-8
            rel = num / den
            return rel if rel.ndim == 0 else rel.mean()

        elif self.loss_type == 'rel_lp':
            num = (torch.abs(pred - target) ** self.p).mean(dim=self.reduce_dims)
            den = (torch.abs(target) ** self.p).mean(dim=self.reduce_dims) + 1e-8
            rel = num / den
            return rel if rel.ndim == 0 else rel.mean()


class SpatialGradientLoss3D(nn.Module):
    def __init__(self, filter_type='sobel', loss_type='l1', reduce_dims=None):
        """
        Spatial gradient-based loss for 3D data.

        Args:
            filter_type: one of ['sobel', 'scharr', 'central', 'laplacian']
            loss_type: one of ['l1', 'mse', 'rel_l1', 'rel_mse']
            reduce_dims: tuple of dims to reduce (only used for relative losses)
                         default: (2, 3, 4, 5) → reduce over (N, D, H, W)
        Return: loss as a scalar
        """
        super().__init__()
        self.filter_type = filter_type.lower()
        self.loss_type = loss_type.lower()

        # set default reduction dims
        self.reduce_dims = reduce_dims if reduce_dims is not None else (1, 2, 3, 4)

        assert self.loss_type in ['l1', 'mse', 'rel_l1', 'rel_mse']
        assert self.filter_type in ['sobel', 'scharr', 'central', 'laplacian']

        self.kernels = get_kernels_3d(self.filter_type)  # shape (N, 1, 3, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: shape (B, C, D, H, W)
        Applies same spatial filters to each channel independently.
        """
        assert pred.shape == target.shape, "Shape mismatch"
        B, C, D, H, W = pred.shape
        N = self.kernels.shape[0]  # number of directional filters
        kernels = self.kernels.to(pred.device, pred.dtype)  # (N,1,3,3,3)

        # Flatten (B, C) → (B*C, 1, D, H, W)
        pred_ = pred.flatten(0, 1).contiguous().unsqueeze(dim=1)
        target_ = target.flatten(0, 1).contiguous().unsqueeze(dim=1)
        # print("After flatten: ", pred_.shape, target_.shape)

        # Apply gradient filters
        grad_pred = F.conv3d(pred_, kernels, padding=1)    # (B*C, N, D, H, W)
        grad_target = F.conv3d(target_, kernels, padding=1)

        # Compute loss
        if self.loss_type == 'l1':
            return F.l1_loss(grad_pred, grad_target)

        elif self.loss_type == 'mse':
            return F.mse_loss(grad_pred, grad_target)

        elif self.loss_type == 'rel_l1':
            num = torch.abs(grad_pred - grad_target).mean(dim=self.reduce_dims)      # (B, C)
            den = torch.abs(grad_target).mean(dim=self.reduce_dims) + 1e-8
            rel = num / den
            return rel.mean()

        elif self.loss_type == 'rel_mse':
            num = ((grad_pred - grad_target) ** 2).mean(dim=self.reduce_dims)  # (B, C)
            den = (grad_target ** 2).mean(dim=self.reduce_dims) + 1e-8
            rel = num / den
            return rel.mean()


class BaseSingleFieldLoss3D(nn.Module):
    def __init__(self, pseudo_3d=False):
        """
        Base class for single-field loss on 3D data.

        Args:
            pseudo_3d: if True, flatten Z into batch and apply 2D loss
        """
        super().__init__()
        self.pseudo_3d = pseudo_3d

    def reshape(self, x):
        """
        Reshape input for loss computation.

        If pseudo_3d=True, reshapes (B, T, X, Y, Z) to (B*T*Z, 1, X, Y)
        Else, keeps as is.
        """
        B, T, X, Y, Z = x.shape
        if self.pseudo_3d:
            return rearrange(x, 'b t x y z -> (b z t) 1 x y')
        else:
            return x

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        """
        preds, trues: (B, T, X, Y, Z)
        """
        return self.loss_fn(self.reshape(preds), self.reshape(trues))

    def loss_fn(self, x, y):
        raise NotImplementedError("Subclasses must implement this method.")


class BaseMultiFieldLoss3D(nn.Module):
    def __init__(self, mode='both', pseudo_3d=False):
        """
        Base class for multi-field loss on 3D data.

        Args:
            mode: 'both', 'pressure', or 'saturation'
            pseudo_3d: if True, flatten Z into batch and apply 2D loss
        """
        super().__init__()
        assert mode in ['both', 'pressure', 'saturation']
        self.mode = mode
        self.pseudo_3d = pseudo_3d

    def reshape(self, x):
        B, T, F, X, Y, Z = x.shape
        if self.pseudo_3d:
            # (B, T, F, X, Y, Z) → (B, Z, T, F, X, Y) → (B*Z, 1, X, Y)
            return rearrange(x, 'b t f x y z -> (b t z) f x y')
        else:
            # (B, T, F, X, Y, Z) → (B, T*F, X, Y, Z)
            return rearrange(x, 'b t f x y z -> b (t f) x y z')

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        p_preds, s_preds = torch.chunk(preds, 2, dim=2) # 
        p_trues, s_trues = torch.chunk(trues, 2, dim=2)

        losses = []
        if self.mode in ['both', 'pressure']:
            losses.append(self.loss_fn(self.reshape(p_preds), self.reshape(p_trues)))
        if self.mode in ['both', 'saturation']:
            losses.append(self.loss_fn(self.reshape(s_preds), self.reshape(s_trues)))

        return sum(losses) / len(losses)

    def loss_fn(self, x, y):
        raise NotImplementedError("Subclasses must implement this method.")
        

@register_multifield_loss("pixel")
class MultiFieldPixelWiseLoss(BaseMultiFieldLoss3D):
    def __init__(self, loss_type='l1', p=2.0, reduce_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.pixel_loss = PixelWiseLoss3D(loss_type, p, reduce_dims)
    
    def loss_fn(self, x, y):
        return self.pixel_loss(x, y)


@register_multifield_loss("gradient")
class MultiFieldGradientLoss(BaseMultiFieldLoss3D):
    def __init__(self, filter_type='sobel', loss_type='l1', reduce_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.grad_loss = SpatialGradientLoss3D(filter_type, loss_type, reduce_dims)

    def loss_fn(self, x, y):
        return self.grad_loss(x, y)


@register_multifield_loss("ssim")
class MultiFieldSSIMLoss(BaseMultiFieldLoss3D):
    def __init__(self, window_size=7, max_val=1.0, reduction='mean', **kwargs):
        self.window_size = window_size
        self.max_val = max_val
        self.reduction = reduction
        super().__init__(**kwargs)
    
    def loss_fn(self, x, y):
        return kornia.losses.ssim3d_loss(
            x, y, 
            window_size=self.window_size, 
            max_val=self.max_val, 
            reduction=self.reduction
        )


@register_multifield_loss("perceptual")
class MultiFieldPerceptualLoss(BaseMultiFieldLoss3D):
    def __init__(self, device, num_img=16, **kwargs):
        kwargs['pseudo_3d'] = True
        super().__init__(**kwargs)
        self.perceptual = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device)
        self.num_img = num_img

    def loss_fn(self, x, y):
        idx = torch.randperm(x.size(0))[:self.num_img]
        return self.perceptual(x[idx], y[idx])


@register_singlefield_loss("pixel")
class SingleFieldPixelWiseLoss(BaseSingleFieldLoss3D):
    def __init__(self, loss_type='l1', p=2.0, reduce_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.pixel_loss = PixelWiseLoss3D(loss_type, p, reduce_dims)
    
    def loss_fn(self, x, y):
        return self.pixel_loss(x, y)


@register_singlefield_loss("gradient")
class SingleFieldGradientLoss(BaseSingleFieldLoss3D):
    def __init__(self, filter_type='sobel', loss_type='l1', reduce_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.grad_loss = SpatialGradientLoss3D(filter_type, loss_type, reduce_dims)
    
    def loss_fn(self, x, y):
        return self.grad_loss(x, y)


@register_singlefield_loss("ssim")
class SingleFieldSSIMLoss(BaseSingleFieldLoss3D):
    def __init__(self, window_size=7, max_val=1.0, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = lambda x, y: kornia.losses.ssim3d_loss(
            x, y, window_size=window_size, max_val=max_val, reduction=reduction
        )


@register_singlefield_loss("perceptual")
class SingleFieldPerceptualLoss(BaseSingleFieldLoss3D):
    def __init__(self, device, num_img=16, **kwargs):
        kwargs['pseudo_3d'] = True
        super().__init__(**kwargs)
        self.perceptual = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device)
        self.num_img = num_img

    def loss_fn(self, x, y):
        idx = torch.randperm(x.size(0))[:self.num_img]
        return self.perceptual(x[idx], y[idx])
