from .adversarial_loss import AdversarialCriterions, PatchDiscriminator, PatchAdversarialLoss

from .layers import (
    get_activation, 
    PixelDropout, 
    ADN, 
    MLP,
    Convolution, 
    Upsample, 
    Downsample, 
    ResBlock, 
    AttentionBlock, 
    LocationEmbedding
)

from .swintr import (
    WindowAttention, 
    SwinTransformerBlock, 
    SpatioTemporalSwinTransformer
)

from .lpips import LPIPS

from .modules import (
    EMAQuantizer, 
    VectorQuantizer,
    PerceptualLoss, 
    Encoder, 
    Decoder, 
    VariationalEncoder, 
    UNet
)

from .pretrained_networks import (
    squeezenet, alexnet, vgg16, resnet
)

from .trainer import GeneralTrainer

from .losses import (
    BaseMultiFieldLoss3D, 
    BaseSingleFieldLoss3D, 
    MultiFieldPixelWiseLoss, 
    MultiFieldGradientLoss, 
    MultiFieldSSIMLoss, 
    MultiFieldPerceptualLoss,
    SingleFieldPixelWiseLoss, 
    SingleFieldGradientLoss, 
    SingleFieldSSIMLoss, 
    SingleFieldPerceptualLoss
)

from .scheduler import (
    ScalarScheduler, 
    MagnitudeMatchScheduler
)

from .registry import (
    get_multifield_loss, 
    get_singlefield_loss,
    register_singlefield_loss,
    register_multifield_loss, 
    MULTIFIELD_LOSS_REGISTRY, 
    SINGLEFIELD_LOSS_REGISTRY
)

from .utils import (plot0, plot1)

from .functions import (compute_paddings_for_conv, 
                        compute_paddings_for_deconv, 
                        calculate_grain_factors, 
                        calculate_factors_product, 
                        calculate_factors_collect)