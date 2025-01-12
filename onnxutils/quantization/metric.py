import torch


def mse_kernel(gt, pred, reduction='none'):
    assert gt.shape == pred.shape
    if gt.dim() < 2:
        gt = gt.reshape(-1, 1)
        pred = pred.reshape(-1, 1)

    gt = gt.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    metrics = torch.pow(gt - pred, 2).mean(dim=-1)

    if reduction == 'none':
        return metrics
    if reduction == 'max':
        return metrics.max()
    if reduction == 'min':
        return metrics.min()
    if reduction == 'mean':
        return metrics.mean()
    if reduction == 'sum':
        return metrics.sum()
    raise NotImplementedError


def cosine_kernel(gt, pred, reduction='none'):
    assert gt.shape == pred.shape
    if gt.dim() < 2:
        gt = gt.reshape(-1, 1)
        pred = pred.reshape(-1, 1)

    gt = gt.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    metrics = torch.cosine_similarity(gt, pred, dim=-1)

    if reduction == 'none':
        return metrics
    if reduction == 'max':
        return metrics.max()
    if reduction == 'min':
        return metrics.min()
    if reduction == 'mean':
        return metrics.mean()
    if reduction == 'sum':
        return metrics.sum()
    raise NotImplementedError


def snr_kernel(gt, pred, eps=1e-7, reduction='none'):
    assert gt.shape == pred.shape
    if gt.dim() < 2:
        gt = gt.reshape(-1, 1)
        pred = pred.reshape(-1, 1)

    gt = gt.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    metrics = (
        torch.pow(gt - pred, 2) / (torch.pow(gt, 2) + eps)
    ).sum(dim=-1)

    if reduction == 'none':
        return metrics
    if reduction == 'max':
        return metrics.max()
    if reduction == 'min':
        return metrics.min()
    if reduction == 'mean':
        return metrics.mean()
    if reduction == 'sum':
        return metrics.sum()
    raise NotImplementedError


def compute_metric(gt, pred, kernel_fn, *args, **kwargs):
    assert type(gt) is type(pred)

    if isinstance(gt, torch.Tensor):
        return kernel_fn(gt, pred, *args, **kwargs)
    if isinstance(gt, (tuple, list)):
        return [
            compute_metric(a, b, kernel_fn, *args, **kwargs)
            for a, b in zip(gt, pred)
        ]
    if isinstance(gt, dict):
        return {
            k: kernel_fn(gt.get(k, None), pred.get(k, None), *args, **kwargs)
            for k in gt.keys() | pred.keys()
        }
