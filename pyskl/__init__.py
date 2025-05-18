# Copyright (c) OpenMMLab. All rights reserved.
from packaging.version import parse as parse_version

try:
    import mmcv
    _mmcv_version = parse_version(mmcv.__version__)
    if _mmcv_version < parse_version('1.3.6'):
        raise RuntimeError(
            f'MMCV>="1.3.6" is required, but {mmcv.__version__} is installed')
except ImportError:
    mmcv = None

from .version import __version__


__all__ = ['__version__']
