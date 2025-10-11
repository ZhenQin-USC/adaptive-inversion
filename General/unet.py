import torch
import torch.nn as nn
from os.path import join
from typing import Callable, List, Optional, Sequence, Tuple, Union


class ConvLSTM3DCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, 
                 kernel_size=3, padding=1, padding_mode='zeros', bias=True, 
                 ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM3DCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.bias = bias

        self.input_conv = nn.Conv3d(in_channels=self.input_dim,
                                    out_channels=4*self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=padding,
                                    padding_mode=self.padding_mode,
                                    stride=(1,1,1),
                                    bias=self.bias)
        
        self.recurrent_conv = nn.Conv3d(in_channels=self.hidden_dim, 
                                        out_channels=4*self.hidden_dim,
                                        kernel_size=self.kernel_size, 
                                        padding=padding,
                                        padding_mode=self.padding_mode,
                                        stride=(1,1,1),
                                        bias=False)
        
        self.recurrent_activation = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, inputs, states):
        h_tm1, c_tm1 = states

        x = self.input_conv(inputs)
        x_i, x_f, x_c, x_o = torch.split(x, self.hidden_dim, dim=1)

        h = self.recurrent_conv(h_tm1)
        h_i, h_f, h_c, h_o = torch.split(h, self.hidden_dim, dim=1)
        
        f = self.recurrent_activation(x_f + h_f)
        i = self.recurrent_activation(x_i + h_i)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, c

    def init_hidden(self, batch_size, input_size):
        depth, height, width = input_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, 
                            device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, 
                            device=self.input_conv.weight.device))


class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, 
                 act_layer: Callable = nn.GELU, stride=1, padding_mode='zeros', 
                 norm_type='group', num_groups=None):

        super(Conv_block, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode)
        if norm_type == 'batch': 
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = out_channels
            self.norm = nn.GroupNorm(num_groups=num_groups, 
                                     num_channels=out_channels)
        self.nonlinear = act_layer()

    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.nonlinear(x)


class ResConv_block(nn.Module):

    def __init__(self, in_channels, kernel_size=3, padding=1, 
                 stride=1, padding_mode='zeros', norm_type='group', num_groups=None):
        
        super(ResConv_block, self).__init__()

        self.conv0 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               padding_mode=padding_mode)
        
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               padding_mode=padding_mode)

        if norm_type == 'batch': 
            self.norm0 = nn.BatchNorm3d(in_channels)
            self.norm1 = nn.BatchNorm3d(in_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = in_channels
            self.norm0 = nn.GroupNorm(num_groups=num_groups, 
                                      num_channels=in_channels)
            self.norm1 = nn.GroupNorm(num_groups=num_groups, 
                                      num_channels=in_channels)
        
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        a = self.nonlinear(self.norm0(a))
        a = self.conv1(a)
        y = self.norm1(a)
        return x + y


class Deconv_block(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size=3, padding=0, 
                 stride=1, padding_mode='reflect', norm_type='group', num_groups=None):
        super(Deconv_block, self).__init__()
        if isinstance(scale_factor, list):
            scale_factor = tuple(scale_factor)
        elif not isinstance(scale_factor, (int, tuple)):
            raise ValueError("scale_factor must be an int, tuple, or list.")

        self.deconv = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              padding_mode=padding_mode)
        if norm_type == 'batch': 
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = out_channels
            self.norm = nn.GroupNorm(num_groups=num_groups, 
                                     num_channels=out_channels)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(self.conv(x))
        return self.nonlinear(x)


class Encoder(nn.Module):
    def __init__(self, 
                 ninputs, 
                 filters=16, 
                 norm_type='group', 
                 num_groups=4, 
                 kernel_size=3, 
                 padding=1, 
                 strides=None):
        
        super(Encoder, self).__init__()

        if strides == None:
            stride0, stride1 = 2, 2
        else:
            stride0, stride1 = strides

        self.conv = nn.Conv3d(ninputs, 1*filters, kernel_size=kernel_size, padding=padding, stride=1)
        
        self.conv0 = Conv_block(
            in_channels=1*filters, out_channels=2*filters, kernel_size=kernel_size, 
            padding=padding, stride=stride0, norm_type=norm_type, num_groups=num_groups)
        
        self.conv1 = Conv_block(
            in_channels=2*filters, out_channels=2*filters, kernel_size=kernel_size, 
            padding=padding, stride=1, norm_type=norm_type, num_groups=num_groups)
        
        self.conv2 = Conv_block(
            in_channels=2*filters, out_channels=4*filters, kernel_size=kernel_size, 
            padding=padding, stride=stride1, norm_type=norm_type, num_groups=num_groups)
        
        self.conv3 = Conv_block(
            in_channels=4*filters, out_channels=4*filters, kernel_size=kernel_size, 
            padding=padding, stride=1, norm_type=norm_type, num_groups=num_groups)


    def forward(self, inputs):
        x0 = self.conv(inputs)
        x1 = self.conv0(x0)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return [x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, noutputs, 
                 filters=16, 
                 norm_type='group', 
                 num_groups=4, 
                 kernel_size=3, 
                 padding=1, 
                 with_control=False, 
                 strides=None):
        
        super(Decoder, self).__init__()
        if strides == None:
            stride1, stride0 = 2, 2
        else:
            stride1, stride0 = strides
            
        self.with_control = with_control 
        if self.with_control:
            coeff=3
        else:
            coeff=2

        self.deconv4 = Deconv_block(
            1, in_channels=coeff*4*filters, out_channels=4*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv3 = Deconv_block(
            stride0, in_channels=coeff*4*filters, out_channels=2*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv2 = Deconv_block(
            1, in_channels=coeff*2*filters, out_channels=2*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv1 = Deconv_block(
            stride1, in_channels=coeff*2*filters, out_channels=1*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.conv = nn.Conv3d(1*filters, noutputs, kernel_size=kernel_size, padding=padding, stride=1)
        
    def forward(self, x, x_enc, u_enc):
        x1, x2, x3, x4 = x_enc
        if self.with_control:
            u1, u2, u3, u4 = u_enc
            x4_ = self.deconv4(torch.cat(( x,  x4, u4), dim=1))
            x3_ = self.deconv3(torch.cat((x4_, x3, u3), dim=1))
            x2_ = self.deconv2(torch.cat((x3_, x2, u2), dim=1))
            x1_ = self.deconv1(torch.cat((x2_, x1, u1), dim=1))
        else:          
            x4_ = self.deconv4(torch.cat(( x,  x4), dim=1))
            x3_ = self.deconv3(torch.cat((x4_, x3), dim=1))
            x2_ = self.deconv2(torch.cat((x3_, x2), dim=1))
            x1_ = self.deconv1(torch.cat((x2_, x1), dim=1))
        outputs = self.conv(x1_)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, num_resblocks: int, filters: int, kernel_size, padding, norm_type, num_groups):
        super().__init__()
        self.filters = filters
        self.num_resblocks = num_resblocks
        self.resnet_config = {
            "kernel_size": kernel_size,
            "padding": padding,
            "norm_type": norm_type,
            "num_groups": num_groups
        }

        self.resnet = self._create_residual_block()

    def _create_residual_block(self):
        resnet = []
        for _ in range(self.num_resblocks):
            resnet.append(ResConv_block(self.filters, **self.resnet_config))
        return nn.Sequential(*resnet)

    def forward(self, x):
        return self.resnet(x)


class Recurrent_RUNet(nn.Module):
    def __init__(self, 
                 filters: int, 
                 units: List[int] = [10, 2, 2], 
                 with_control: bool = False, 
                 with_states: bool = True,
                 norm_type: str = 'group', 
                 num_groups: int = 4, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 strides: List[int] = [2, 2], 
                 ):
        
        super(Recurrent_RUNet, self).__init__()
        ninputs, ncontrols, noutputs = units
        self.with_control = with_control
        self.with_states = with_states
        self.x_encoder = Encoder(ninputs, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)
        
        self.u_encoder = Encoder(ncontrols, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)
        
        self.resnet0 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet1 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet2 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet3 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        if with_states:
            self.x_conv  = Conv_block(in_channels=8*filters, out_channels=4*filters, 
                                      kernel_size=kernel_size, padding=padding, stride=1,
                                      norm_type=norm_type, num_groups=num_groups)
        else:
            self.x_conv  = Conv_block(in_channels=4*filters, out_channels=4*filters, 
                                      kernel_size=kernel_size, padding=padding, stride=1,
                                      norm_type=norm_type, num_groups=num_groups)

        self.convlstm = ConvLSTM3DCell(4*filters, 4*filters, kernel_size=kernel_size, padding=padding)
        self.decoder  = Decoder(noutputs, filters=filters, with_control=with_control, 
                                kernel_size=kernel_size, padding=padding, 
                                norm_type=norm_type, num_groups=num_groups, strides=strides)

    def forward(self, contrl, _states, _static):
        B, steps, _, Nx, Ny, Nz = contrl.size()
        output_seq = []
        states = torch.cat((_states, _static), dim=1)
        # Encoder:
        x_enc1, x_enc2, x_enc3, x_enc4 = self.x_encoder(states)
        # ResNet After Encoder:
        x4 = self.resnet0(x_enc4)
        x4 = self.resnet1(x4)
        hidden_states = [x4, x4]
        
        # ConvLSTM:
        for step in range(steps):
            _contrl = contrl[:,step,...]
            u_enc1, u_enc2, u_enc3, u_enc4 = self.u_encoder(_contrl)
            if self.with_states:
                x = self.x_conv(torch.cat((u_enc4, x4), dim=1))
            else:
                x = self.x_conv(u_enc4)
            x, c = self.convlstm(x, states=hidden_states)
            hidden_states = [x, c]

            # ResNet Before Decoder:
            x = self.resnet2(x)
            x = self.resnet3(x)
            out = self.decoder(x, 
                               [x_enc1, x_enc2, x_enc3, x_enc4],
                               [u_enc1, u_enc2, u_enc3, u_enc4]
                               )
            output_seq.append(out)

        outputs = torch.stack(output_seq, dim=1)
        return outputs, outputs


class RecurrentUNet2(nn.Module):
    def __init__(self, 
                 filters: int, 
                 units: List[int] = [10, 2, 2], 
                 with_control: bool = False, 
                 with_states: bool = True,
                 norm_type: str = 'group', 
                 num_groups: int = 4, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 strides: List[int] = [2, 2], 
                 num_resblocks: int = 2,
             ):
        
        super(RecurrentUNet2, self).__init__()
        nstatic, ncontrl, nstates = units
        self.with_control = with_control
        self.with_states = with_states
        self.resnet_config = {"filters": 4*filters, "kernel_size": kernel_size, "padding": padding, 
                              "norm_type": norm_type, "num_groups": num_groups}
        
        self.scale_m = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(4)])
        self.scale_u = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(4)])

        self.x_encoder = Encoder(nstates, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)
        
        self.m_encoder = Encoder(nstatic, filters=filters, norm_type=norm_type,
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)

        self.u_encoder = Encoder(ncontrl, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)

        self.x_decoder  = Decoder(nstates, filters=filters, with_control=with_control, 
                                  kernel_size=kernel_size, padding=padding, 
                                  norm_type=norm_type, num_groups=num_groups, strides=strides)

        self.x_enc_res = ResidualBlock(num_resblocks, **self.resnet_config)
        self.m_enc_res = ResidualBlock(num_resblocks, **self.resnet_config)
        self.u_enc_res = ResidualBlock(num_resblocks, **self.resnet_config)
        self.x_dec_res = ResidualBlock(num_resblocks, **self.resnet_config)
        
        if with_states:
            self.x_conv  = Conv_block(in_channels=8*filters, out_channels=4*filters, 
                                      kernel_size=kernel_size, padding=padding, stride=1,
                                      norm_type=norm_type, num_groups=num_groups)
        else:
            self.x_conv  = Conv_block(in_channels=4*filters, out_channels=4*filters, 
                                      kernel_size=kernel_size, padding=padding, stride=1,
                                      norm_type=norm_type, num_groups=num_groups)

        self.convlstm = ConvLSTM3DCell(4*filters, 4*filters, kernel_size=kernel_size, padding=padding)

    def forward(self, contrl, states, static):
        B, T, _, Nx, Ny, Nz = contrl.size()

        # States Encoder:
        x_enc1, x_enc2, x_enc3, x_enc4 = self.x_encoder(states) # (B, C, X, Y, Z)
        # Static Encoder:
        m_enc1, m_enc2, m_enc3, m_enc4 = self.m_encoder(static) # (B, C, X, Y, Z)
        # Contrl Encoder:
        u_enc1, u_enc2, u_enc3, u_enc4 = self.parallel_u_encoder(contrl) # (B*T, C, X, Y, Z)

        # ResNet:
        x4 = self.x_enc_res(x_enc4)
        m4 = self.m_enc_res(m_enc4)
        u4 = self.u_enc_res(u_enc4)
        u4 = u4.reshape(B, T, *u4.shape[1:]) # (B, T, C, X, Y, Z)
        hidden_states = [x4, m4]
        
        # ConvLSTM:
        latent_seq = []
        for step in range(T):
            if self.with_states:
                x = self.x_conv(torch.cat((u4[:, step], m4), dim=1))
            else:
                x = self.x_conv(u4[:, step])

            x, c = self.convlstm(x, states=hidden_states)
            hidden_states = [x, c]
            latent_seq.append(x)

        latent_seq = torch.stack(latent_seq, dim=1)  # (B, T, C, X, Y, Z)

        outputs = self.parallel_x_decoder(latent_seq,  # (B, T, C, X, Y, Z)
                                          [m_enc1, m_enc2, m_enc3, m_enc4],
                                          [u_enc1, u_enc2, u_enc3, u_enc4])
        
        return outputs, outputs

    def parallel_x_decoder(self, latent_seq, m_encs, u_encs):
        B, T, C, X, Y, Z = latent_seq.shape

        x = latent_seq.reshape(B * T, C, X, Y, Z)

        m_encs_scaled = [
            torch.repeat_interleave((enc * self.scale_m[i]), repeats=T, dim=0) for i, enc in enumerate(m_encs)
            ]
        #[(enc * self.scale_m[i]).unsqueeze(1).repeat(1, T, *([1] * (enc.ndim - 1))).reshape(B * T, *enc.shape[1:]) for i, enc in enumerate(m_encs)] # 

        u_encs_scaled = [
            enc * self.scale_u[i]
            for i, enc in enumerate(u_encs)
        ]

        x = self.x_dec_res(x) # (B*T, C, X, Y, Z)

        out = self.x_decoder(x, m_encs_scaled, u_encs_scaled) # (B*T, C, X, Y, Z)

        output_seq = out.reshape(B, T, *out.shape[1:])
        return output_seq

    def parallel_u_encoder(self, contrl):
        B, T, C, X, Y, Z = contrl.shape

        x = contrl.reshape(B * T, C, X, Y, Z)
        u_encs = self.u_encoder(x)

        return u_encs # tuple(enc.reshape(B, T, *enc.shape[1:]) for enc in u_encs)
