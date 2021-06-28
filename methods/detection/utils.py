import torch
import torch.distributed as dist


def collate_fn(batch):
    return list(zip(*batch))


def build_coco_compliant_batch(image_and_target_batch):
    images, bboxes = zip(*image_and_target_batch)

    def _get_coco_target(bboxes):
        n_boxes = len(bboxes)
        boxes = [[x0, y0, x1, y1] for y0, x0, y1, x1 in bboxes] if n_boxes else [[]]
        shape = (n_boxes,) if n_boxes else (1, 0)
        return {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.ones(shape, dtype=torch.int64),  # there is only one class
            'iscrowd': torch.zeros(shape, dtype=torch.int64)  # suppose all instances are not crowd
        }

    targets = [_get_coco_target(b) for b in bboxes]
    return images, targets


def check_empty_images(targets):
    if targets[0]['boxes'].is_cuda:
        device = targets[0]['boxes'].get_device()
    else:
        device = torch.device("cpu")

    for target in targets:
        if target['boxes'].nelement() == 0:
            target['boxes'] = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32, device=device)
            target['labels'] = torch.zeros((1,), dtype=torch.int64, device=device)
            target['iscrowd'] = torch.zeros((1,), dtype=torch.int64, device=device)

    return targets


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
