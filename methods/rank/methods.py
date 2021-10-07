import torch
from torch import nn
import torch.nn.functional as F
import torchsort

from spacecutter.losses import cumulative_link_loss


def spearmanr(pred, target, **kw):
    device = pred.device
    pred = torchsort.soft_rank(pred.cpu(), **kw).to(device)
    target = torchsort.soft_rank(target.cpu(), **kw).to(device)
    pred = pred - pred.mean()
    pred = pred / (pred.norm() + 1e-8)
    target = target - target.mean()
    target = target / (target.norm() + 1e-8)
    return (pred * target).sum()
    

def _score_metrics(scores, cfg, targets=None):
    s = scores.flatten()

    # KL divergence against uniform distribution (non-differentiable)
    # bins, eps = 10, 10**-6
    bins = cfg.optim.loss.kl_div.bins
    eps = cfg.optim.loss.kl_div.eps
    qi = torch.histc(s, bins=bins, min=0, max=1) + eps
    qi /= qi.sum()
    # pi = 1. / bins;   kl_div = pi * torch.log(pi / qi)  OR:
    kl_div = (- torch.log(qi * bins) / bins).sum()

    # spreading loss: minimum when the whole [0, 1] range is occupied
    t = cfg.optim.loss.spread.temperature
    spread_loss = 1 - torch.mean(s * F.softmax(s / t, dim=0)) + torch.mean(s * F.softmin(s / t, dim=0))

    ret = {
        'rank/kl_div': kl_div,
        'rank/spread_loss': spread_loss,
    }

    if targets is not None:
        scores = scores.reshape(1, -1)
        targets = targets.reshape(1, -1)
        ret['rank/spearman'] = spearmanr(scores, targets, regularization_strength=0.1)
    
    return ret


def simple_regression(sample, model, device, cfg):
    """ Computes simple regression loss, returns losses and scores.

    Args:
        sample (tuple): a tuple containing a batch of images and targets
        model (nn.Module): a model returning a score
        device (torch.device): device to use
        cfg (DictConfig): an OmegaConf config dictionary

    Returns:
        dict: losses and metrics
        tensor: scores of the images
    """
    num_classes = 7

    x, y = sample
    x = x.to(device)
    # y = ((y - 1) / (num_classes - 1)).to(device)  # map classes to increasing scores in [0, 1]
    y = (y / num_classes).to(device)  # map classes to increasing scores in [0, 1]

    scores = model(x)
    # scores = torch.sigmoid(scores)

    regression_loss = torch.mean((y - scores) ** 2)
    terms = _score_metrics(scores, cfg, targets=y)
    
    loss = regression_loss + \
           cfg.optim.loss.spread.weight * terms['rank/spread_loss']

    metrics = {
        'loss': loss,
        'regression_loss': regression_loss,
        **terms,
    }
    return metrics, scores


def simple_classification(sample, model, device, cfg):
    """ Computes classification loss and map multinomial output to a score.
        Returns losses and scores.

    Args:
        sample (tuple): a tuple containing a batch of images and targets
        model (nn.Module): a model returning a N-class multinomial output
        device (torch.device): device to use
        cfg (DictConfig): an OmegaConf config dictionary

    Returns:
        dict: losses and metrics
        tensor: scores of the images
    """
    x, y = sample
    # x, y = x.to(device), (y - 1).to(device)  # y: [1, 7] --> [0, 6]
    x, y = x.to(device), y.to(device)

    logits = model(x)

    batch_size, n_categories = logits.shape[:2]
    classification_loss = F.cross_entropy(logits, y, reduction='mean')

    pivot_scores = torch.linspace(0, 1, n_categories).to(logits.device).unsqueeze(0)
    scores = (pivot_scores * torch.softmax(logits, dim=1)).sum(axis=1)

    terms = _score_metrics(scores, cfg, targets=y)

    loss = cfg.optim.loss.classification.weight * classification_loss + \
           cfg.optim.loss.spread.weight * terms['rank/spread_loss']

    metrics = {
        'loss': loss,
        'rank/classif_loss': classification_loss,
        **terms
    }

    return metrics, scores


def spearmanr_optimization(sample, model, device, cfg):
    x, y = sample
    x, y = x.to(device), y.to(device)

    logits = model(x)
    scores = torch.sigmoid(logits)

    terms = _score_metrics(scores, cfg, targets=y)

    metrics = {
        'loss': - terms['rank/spearman'],  # maximize spearman corr coef
        **terms
    }

    return metrics, scores


def ordinal_regression(sample, model, device, cfg):
    """ Computes ordinal regression loss using learned thresholds / cutpoints.
        Returns losses and scores.

    Args:
        sample (tuple): a tuple containing a batch of images and targets
        model (nn.Module): an instance of spacecutter.models.OrdinalLogisticModel
        device (torch.device): device to use
        cfg (DictConfig): an OmegaConf config dictionary

    Returns:
        dict: losses and metrics
        tensor: scores of the images
    """
    x, y_true = sample
    x, y_true = x.to(device), y_true.to(device)

    scores = model.predictor(x)
    y_pred = model.link(scores)

    loss = cumulative_link_loss(y_pred, y_true.reshape(-1, 1), class_weights=cfg.optim.class_weights)
    terms = _score_metrics(scores, cfg, targets=y_true)
    metrics = { 'loss': loss, **terms }

    return metrics, scores


def pairwise_unbalanced(sample, model, device, cfg):
    """ Computes pairwise margin loss between all samples in the batch.
        Returns losses and scores.

    Args:
        sample (tuple): a tuple containing a batch of images and targets
        model (nn.Module): a model returning a score
        device (torch.device): device to use
        cfg (DictConfig): an OmegaConf config dictionary

    Returns:
        dict: losses and metrics
        tensor: scores of the images
    """
    x, y = sample
    x, y = x.to(device), y.to(device)

    scores = model(x)
    scores = torch.sigmoid(scores)

    b = scores.shape[0]

    s = scores.reshape(b, 1)
    y = y.reshape(b, 1)

    ds = s - s.T
    dy = y - y.T

    # margin ranking loss (weighted by dy that can be |dy| != 1)
    rank_loss = F.relu(-dy*ds + cfg.optim.loss.rank.margin).mean()

    terms = _score_metrics(scores, cfg, targets=y)

    loss = cfg.optim.loss.rank.weight * rank_loss + \
           cfg.optim.loss.spread.weight * terms['rank/spread_loss']

    metrics = {
        'loss': loss,
        'rank/margin_loss': rank_loss,
        **terms,
    }

    return metrics, scores


def pairwise_balanced(sample, model, device, cfg):
    """ Computes pairwise margin loss between batches of increasing classes.
        Returns losses and scores.

    Args:
        sample (tuple): a tuple containing N batches of images, one for each class in increasing order
        model (nn.Module): a model returning a score
        device (torch.device): device to use
        cfg (DictConfig): an OmegaConf config dictionary

    Returns:
        dict: losses and metrics
        tensor: scores of the images
    """
    if len(sample) == 2:  # patch and labels (TODO FIX collides with N=2)
        x, y = sample
        x, y = x.to(device), y.to(device)
        scores = model(x)
        scores = torch.sigmoid(scores)

        metrics = _score_metrics(scores, cfg, targets=y)
        return metrics, scores

    # tuples
    scores = [model(x.to(device)) for x in sample]
    scores = torch.cat(scores, dim=1)
    scores = torch.sigmoid(scores)

    diffs = scores[:, 1:] - scores[:, :-1]  # diffs = torch.diff(scores)  # needs torch >= 1.8.0
    rank_loss = F.relu(cfg.optim.loss.rank.margin - diffs).sum(axis=1).mean()
    sorted_pct = (diffs >= 0).all(axis=1).float().mean()

    batch_size, num_classes = scores.shape
    targets = torch.arange(num_classes).repeat(batch_size, 1)
    terms = _score_metrics(scores, cfg, targets=targets)

    loss = cfg.optim.loss.rank.weight * rank_loss + \
           cfg.optim.loss.spread.weight * terms['rank/spread_loss']

    metrics = {
        'loss': loss,
        'rank/margin_loss': rank_loss,
        'rank/sorted_pct': sorted_pct,
        **terms,
    }

    return metrics, scores
