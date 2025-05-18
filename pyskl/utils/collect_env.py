# Copyright (c) OpenMMLab. All rights reserved.
"""Utilities to gather environment information without relying on MMCV."""

import platform
import subprocess
import sys
import torch
from collections import OrderedDict

import pyskl


def get_git_hash(digits=7):
    """Get the git hash of the current repo.

    Args:
        digits (int): The number of digits of the hash string. Defaults to 7.

    Returns:
        str: Git commit hash of specified length. ``"unknown"`` if failed.
    """

    cmd = ["git", "rev-parse", "HEAD"]
    try:
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        sha = sha.decode("ascii").strip()
    except Exception:
        sha = "unknown"
    if digits and digits > 0:
        sha = sha[:digits]
    return sha


def collect_env():
    """Collect the environment information."""

    env_info = OrderedDict()
    env_info["sys.platform"] = platform.platform()
    env_info["Python"] = sys.version.replace("\n", "")
    env_info["CUDA available"] = torch.cuda.is_available()
    env_info["pyskl"] = pyskl.__version__ + "+" + get_git_hash(digits=7)

    # PyTorch and CUDA related info
    env_info["PyTorch"] = torch.__version__
    try:
        env_info["PyTorch compiling GPU"] = torch.version.cuda
    except AttributeError:
        env_info["PyTorch compiling GPU"] = "None"

    if env_info["CUDA available"]:
        env_info["GPU"] = ", ".join(
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        )
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            env_info["CUDA_HOME"] = str(CUDA_HOME)
        except Exception:
            env_info["CUDA_HOME"] = "Not Available"
        try:
            nvcc_out = subprocess.check_output(["nvcc", "--version"])
            nvcc_out = nvcc_out.decode("utf-8").strip().split("\n")[-1]
            env_info["NVCC"] = nvcc_out
        except Exception:
            env_info["NVCC"] = "Not Available"
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
