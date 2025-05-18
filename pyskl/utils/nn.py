import torch
import torch.nn as nn

__all__ = [
    'build_activation_layer', 'build_norm_layer', 'ConvModule',
    'constant_init', 'kaiming_init', 'normal_init',
    'load_checkpoint', '_load_checkpoint', '_BatchNorm', 'Swish'
]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def build_activation_layer(cfg):
    if cfg is None:
        return nn.Identity()
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    if hasattr(nn, layer_type):
        cls = getattr(nn, layer_type)
        return cls(**cfg)
    if layer_type.lower() == 'swish':
        return Swish()
    raise KeyError(f'Unsupported activation type {layer_type}')


def build_norm_layer(cfg, num_features, postfix=''):
    cfg = cfg.copy()
    layer_type = cfg.pop('type', 'BN')
    if layer_type in ['BN', 'BN2d']:
        layer = nn.BatchNorm2d(num_features, **cfg)
    elif layer_type == 'BN1d':
        layer = nn.BatchNorm1d(num_features, **cfg)
    elif layer_type == 'BN3d':
        layer = nn.BatchNorm3d(num_features, **cfg)
    else:
        raise KeyError(f'Unsupported norm type {layer_type}')
    return layer_type + postfix, layer


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if bias == 'auto':
            bias = norm_cfg is None
        conv_type = conv_cfg['type'] if conv_cfg else 'Conv2d'
        conv_class = getattr(nn, conv_type)
        self.conv = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if norm_cfg:
            _, self.bn = build_norm_layer(norm_cfg, out_channels)
        else:
            self.bn = None
        self.activate = build_activation_layer(act_cfg) if act_cfg else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, bias)


_BatchNorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _load_checkpoint(filename):
    return torch.load(filename, map_location='cpu')


def load_checkpoint(model, filename, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint
