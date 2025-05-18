import os
import os.path as osp
import shutil
import tempfile
import torch
import torch.distributed as dist

from ..utils import get_dist_info, load as _load, dump as _dump


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    else:
        os.makedirs(tmpdir, exist_ok=True)
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')
    _dump(result_part, part_file)
    dist.barrier()

    if rank == 0:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.extend(_load(part_file))
        part_list = part_list[:size]
        shutil.rmtree(tmpdir)
        return part_list
    else:
        return None


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    rank, _ = get_dist_info()
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)
    results = collect_results(results, len(data_loader.dataset), tmpdir)
    return results
