# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .testing import multi_gpu_test
from ..utils import (cache_checkpoint, get_dist_info, get_root_logger,
                     build_optimizer)

from ..datasets import build_dataloader, build_dataset


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.get('log_level', 'INFO'))

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', True)
    model = DistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    optimizer = build_optimizer(model, cfg.optimizer)
    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)
    max_epochs = cfg.total_epochs
    start_epoch = 0

    if cfg.get('resume_from'):
        checkpoint = torch.load(cfg.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer', {}))
        start_epoch = checkpoint.get('epoch', 0)
    elif cfg.get('load_from'):
        cfg.load_from = cache_checkpoint(cfg.load_from)
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)

    for epoch in range(start_epoch, max_epochs):
        for loader in data_loaders:
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
        model.train()
        for data in data_loaders[0]:
            outputs = model.module.train_step(data, optimizer)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            grad_clip = cfg.optimizer_config.get('grad_clip')
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **grad_clip)
            optimizer.step()

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        ckpt_path = osp.join(work_dir, f'epoch_{epoch + 1}.pth')
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, osp.join(work_dir, 'latest.pth'))

        if validate and (epoch + 1) % eval_cfg.get('interval', 1) == 0:
            outputs = multi_gpu_test(model, val_dataloader, eval_cfg.get('tmpdir'))
            rank, _ = get_dist_info()
            if rank == 0:
                eval_params = {
                    k: v
                    for k, v in eval_cfg.items()
                    if k not in [
                        'interval', 'tmpdir', 'start', 'save_best', 'rule',
                        'by_epoch', 'broadcast_bn_buffers'
                    ]
                }
                eval_res = val_dataset.evaluate(outputs, **eval_params)
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')

    dist.barrier()
    time.sleep(2)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            ckpt_paths = [x for x in os.listdir(work_dir) if 'best' in x and x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths]
                best_ckpt_path = osp.join(work_dir, ckpt_paths[np.argmax(epoch_ids)])
            else:
                best_ckpt_path = osp.join(work_dir, ckpt_paths[0])

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []
        if test['test_last']:
            names.append('last')
            ckpts.append(osp.join(work_dir, 'latest.pth'))
        if test['test_best']:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                checkpoint = torch.load(ckpt, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)

            outputs = multi_gpu_test(model, test_dataloader, cfg.get('evaluation', {}).get('tmpdir'))
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(work_dir, f'{name}_pred.pkl')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch',
                        'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')
