import pytorch_lightning as pl
import torch


def set_seed(seed=42, reproducible=True):
    """
    Sets seeds for reproducibility or profiling performance.

    Args:
        seed (int): Seed to use.
        reproducible (bool): If True, ensures exact reproducibility (slower).
                             If False, better performance.
    """
    pl.seed_everything(seed, workers=reproducible)
    torch.cuda.manual_seed_all(seed)

    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
