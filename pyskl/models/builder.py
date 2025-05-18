# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmcv.cnn import MODELS as MMCV_MODELS
except ImportError:  # pragma: no cover
    MMCV_MODELS = None
from ..utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg):
    """Build recognizer."""
    return RECOGNIZERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_model(cfg):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in RECOGNIZERS:
        return build_recognizer(cfg)
    raise ValueError(f'{obj_type} is not registered')
