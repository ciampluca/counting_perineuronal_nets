import torch


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