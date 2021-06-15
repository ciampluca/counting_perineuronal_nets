import collections
import os
from pathlib import Path
import re

import numpy as np
import torch
import torch.distributed as dist


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def unbatch(batches):
    for batch in batches:
        yield from zip(*batch)


class CheckpointManager:

    COMMON_MODES={
        'segm/weighted_bce_loss': 'min',
        'segm/soft_dice': 'min',
        'segm/soft_jaccard': 'max',
        'pdet/average_precision': 'max',
        'segm/dice': 'max',
        'segm/jaccard': 'max',
        'pdet/precision': 'max',
        'pdet/recall': 'max',
        'pdet/f1_score': 'max',
        'count/err': 'ignore',
        'count/mae': 'min',
        'count/mse': 'min',
        'count/mare': 'min',
        'count/game-0': 'min',
        'count/game-1': 'min',
        'count/game-2': 'min',
        'count/game-3': 'min',
        'count/game-4': 'min',
        'count/game-5': 'min',
    }

    def __init__(
        self,
        ckpt_dir,
        ckpt_format=None,
        current_best={},
        metric_modes=None
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_format = ckpt_format if ckpt_format else self._default_ckpt_format
        self.current_best = collections.defaultdict(lambda: None, current_best)
        self.metric_modes = metric_modes if metric_modes else self.COMMON_MODES
    
    def save(self, ckpt, metrics, epoch):
        
        ckpt_path = self.ckpt_dir / f'ckpt_e{epoch}.pth'

        for metric, metric_info in metrics.items():
            value = metric_info.get('value', None)
            threshold = metric_info.get('threshold', None)

            mode = self.metric_modes.get(metric, 'ignore')
            if mode == 'ignore':
                continue
            
            cur_best = self.current_best[metric] and self.current_best[metric].get('value', None)
            is_new_best = cur_best is None or ((value < cur_best) if mode == 'min' else (value > cur_best))
            if is_new_best:
                if not ckpt_path.exists():  # save ckpt if not already saved
                    torch.save(ckpt, ckpt_path)

                # create a link indicating a best ckpt
                best_metric_ckpt_name = self.ckpt_format(metric, value, threshold, epoch)
                best_metric_ckpt_path = self.ckpt_dir / best_metric_ckpt_name
                best_metric_ckpt_path.unlink(missing_ok=True)
                best_metric_ckpt_path.symlink_to(ckpt_path.name)

                # update current best
                self.current_best[metric] = {'value': value, 'threshold': threshold, 'epoch': epoch}

        self.house_keeping()  # deletes orphan checkpoints
        return dict(self.current_best)
    
    @staticmethod
    def _default_ckpt_format(metric_name, metric_value, metric_thr, epoch):
        metric_name = metric_name.replace('/', '-')
        return f'best_model_metric_{metric_name}.pth'

    def house_keeping(self):
        maybe_ckpts = self.ckpt_dir.glob('ckpt_e*.pth')

        ckpt_re = re.compile(r'ckpt_e\d+\.pth')
        ckpts = filter(lambda p: ckpt_re.match(p.name), maybe_ckpts)
        ckpts = map(lambda p: p.resolve(), ckpts)
        ckpts = set(list(ckpts))

        symlinks = self.ckpt_dir.glob('*.pth')
        symlinks = filter(lambda p: p.is_symlink(), symlinks)
        symlinks = map(lambda p: p.resolve(), symlinks)
        symlinks = set(list(symlinks))

        unused_ckpts = ckpts - symlinks
        for unused_ckpt in unused_ckpts:
            unused_ckpt.unlink()


if __name__ == '__main__':

    epoch_metrics = [
        {
            'segm/weighted_bce_loss': {'value': 1, 'threshold': None},
            'segm/jaccard': {'value': 3, 'threshold': 0.5},
            'count/err': {'value': -3, 'threshold': 0.7}
        },
        {
            'segm/weighted_bce_loss': {'value': 0.5, 'threshold': None},
            'segm/jaccard': {'value': 4, 'threshold': 0.3},
            'count/err': {'value': -2, 'threshold': 0.9}
        },
        {
            'segm/weighted_bce_loss': {'value': 1, 'threshold': None},
            'segm/jaccard': {'value': 5, 'threshold': 0.5},
            'count/err': {'value': -7, 'threshold': 0.3}
        }
    ]

    ckpt_dir = Path('trash/ckpt_man_test')
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_man = CheckpointManager(ckpt_dir)

    for epoch, metrics in enumerate(epoch_metrics, start=1):
        ckpt_man.save({}, metrics, epoch)
    
    print(ckpt_man.current_best)

    import pdb; pdb.set_trace()