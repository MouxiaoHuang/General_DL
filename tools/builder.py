import math
import torch
import warnings
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
import models.fas as fas
import datasets as ds
from torch.utils.data import DataLoader
from .sampler import BalanceSampler, DistBalanceSampler, SwitchSampler


def build_models(cfg, **kwargs):
    """build model from config dict.

    Args:
        cfg (dict): the config dict of model.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    model_type = cfg_.pop('type')
    model = eval(f'fas.{model_type}')(**cfg_, **kwargs)
    return model


def build_datasets(cfg, **kwargs):
    """build dataset from config dict.

    Args:
        cfg (dict): the config dict of dataset.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    dataset_type = cfg_.pop('type')
    dataset = eval(f'ds.{dataset_type}')(**cfg_, **kwargs)
    return dataset


def build_dataloaders(cfg, dataset, num_replicas=None, rank=None, **kwargs):
    """build dataloader from config dict.

    Args:
        cfg (dict): the config dict of dataloader.
        dataset (torch.utils.data.Dataset): the dataset to load.
        num_replicas (int, optional): number of replicas for distributed.
        rank (int, optional): current rank.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    sampler = None
    distributed = num_replicas is not None and rank is not None
    if cfg_.pop('shuffle', False):
        if dataset.test_mode:
            warnings.warn('Using BalanceSampler when test mode is True.')
        if distributed:
            sampler = DistBalanceSampler(dataset, num_replicas, rank)
        else:
            sampler_type = cfg_.pop('sampler', 'BalanceSampler')
            sampler = eval(sampler_type)(dataset, cfg['samples_per_gpu'])
    num_gpus = cfg_.pop('num_gpus')
    if distributed:
        batch_size = cfg_.pop('samples_per_gpu')
        num_workers = cfg_.pop('workers_per_gpu')
    else:
        batch_size = cfg_.pop('samples_per_gpu') * num_gpus
        num_workers = cfg_.pop('workers_per_gpu') * num_gpus
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        **cfg_,
        **kwargs)
    return dataloader


def build_optimizers(cfg, model, **kwargs):
    """build optimizer from config dict.

    Args:
        cfg (dict): the config dict of optimizer.
        model (nn.Module): the model to optimize.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    optimizer_type = cfg_.pop('type')
    optimizer = eval(f'opt.{optimizer_type}')(model.parameters(), **cfg_, **kwargs)
    return optimizer


def build_schedulers(cfg, optimizer, **kwargs):
    """build scheduler from config dict.

    Args:
        cfg (dict): the config dict of scheduler.
        optimizer (torch.optim) the torch optimizer.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    scheduler_type = cfg_.pop('type')
    if scheduler_type == 'CosineLR':
        T = cfg_.pop('total_epochs', 50)
        gamma = cfg_.pop('gamma', 0.1)
        lambda1 = lambda epoch: gamma if 0.5 * (1 + math.cos(math.pi * epoch / T)) < gamma else 0.5 * (
                1 + math.cos(math.pi * epoch / T))
        scheduler = lrs.LambdaLR(optimizer, lr_lambda=lambda1, **kwargs)
    else:
        scheduler = eval(f'lrs.{scheduler_type}')(optimizer, **cfg_, **kwargs)
    return scheduler
