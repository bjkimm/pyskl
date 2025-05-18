# Copyright (c) OpenMMLab. All rights reserved.
import warnings

try:
    import mmcv  # type: ignore
except ImportError:
    warnings.warn('mmcv is not installed; some functions may be unavailable.',
                  ImportWarning)
    mmcv = None

from .version import __version__

__all__ = ['__version__']
