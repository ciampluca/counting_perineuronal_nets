import numpy as np

import torch
import torch.distributed as dist


def collate_fn(batch):
    return list(zip(*batch))


def build_coco_compliant_batch(image_and_target_batch, mask=False):
    if mask:
        images, bboxes, labels, mask_segmentations = zip(*image_and_target_batch)
    else:
        images, bboxes, labels = zip(*image_and_target_batch)

    def _get_coco_target(bboxes, labels, mask_segmentations=None, mask=False):
        n_boxes = len(bboxes)
        shape = (n_boxes,) if n_boxes else (1, 0)

        # In case of empty images (i.e, without bbs), we handle them as negative images
        # (i.e., images with only background and no object), creating a fake object that represent the background
        # class and does not affect training
        # https://discuss.pytorch.org/t/torchvision-faster-rcnn-empty-training-images/46935/12
        boxes = [[x0, y0, x1, y1] for y0, x0, y1, x1 in bboxes] if n_boxes else [[0, 1, 2, 3]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # +1 to add the BG class
        labels = torch.as_tensor(labels + 1, dtype=torch.int64) if n_boxes else torch.zeros((1), dtype=torch.int64)
        
        if mask:
            if mask_segmentations.shape[-1] == 0:
                mask_segmentations = np.zeros((*mask_segmentations.shape[:2], 1), dtype=np.int64)
            masks = torch.as_tensor(np.swapaxes(mask_segmentations, 0, 2), dtype=torch.uint8)
            return {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'iscrowd': torch.zeros(shape, dtype=torch.int64)  # suppose all instances are not crowd
            }
            
        return {
            'boxes': boxes,
            'labels': labels,
            'iscrowd': torch.zeros(shape, dtype=torch.int64)  # suppose all instances are not crowd
        }

    if mask:
        targets = [_get_coco_target(b, l, m, mask=mask) for b, l, m in zip(bboxes, labels, mask_segmentations)]
    else:
        targets = [_get_coco_target(b, l, mask=mask) for b, l in zip(bboxes, labels)]
    return images, targets


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
