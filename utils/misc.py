import numpy as np
import os
from collections import defaultdict, deque
import time
import datetime
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
import timeit

import torch
import torch.distributed as dist
import torchvision

import utils.transforms_dmaps as dmap_custom_T
import utils.transforms_bbs as bbox_custom_T
from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame


def get_dmap_transforms(train=False, crop_width=1920, crop_height=1080):
    transforms = []

    if train:
        transforms.append(dmap_custom_T.RandomHorizontalFlip())
        transforms.append(dmap_custom_T.RandomCrop(width=crop_width, height=crop_height))
        transforms.append(dmap_custom_T.PadToResizeFactor())

    if not train:
        transforms.append(dmap_custom_T.PadToResizeFactor(resize_factor=crop_width))

    transforms.append(dmap_custom_T.ToTensor())

    return dmap_custom_T.Compose(transforms)


def get_bbox_transforms(train=False, crop_width=640, crop_height=640, resize_factor=640, min_visibility=0.0):
    transforms = []

    if train:
        transforms.append(bbox_custom_T.RandomHorizontalFlip())
        transforms.append(bbox_custom_T.RandomCrop(width=crop_width, height=crop_height, min_visibility=min_visibility))
    else:
        transforms.append(bbox_custom_T.PadToResizeFactor(resize_factor=resize_factor))

    transforms.append(bbox_custom_T.ToTensor())

    return bbox_custom_T.Compose(transforms)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(data, path, best_model=None, only_last=True):
    if not os.path.exists(path):
        os.makedirs(path)
    epoch = data['epoch']
    if best_model:
        outfile = 'best_model_{}.pth'.format(best_model)
    else:
        if only_last:
            outfile = 'last.pth'
        else:
            outfile = 'checkpoint_epoch_{}.pth'.format(epoch)
    outfile = os.path.join(path, outfile)
    torch.save(data, outfile, _use_new_zipfile_serialization=False)


def normalize(np_arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    minval = np_arr.min()
    maxval = np_arr.max()
    if minval != maxval:
        np_arr -= minval
        np_arr *= (255.0/(maxval-minval))

    return np_arr


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


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


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def check_empty_images(targets):
    if targets[0]['boxes'].is_cuda:
        device = targets[0]['boxes'].get_device()
    else:
        device = torch.device("cpu")

    for target in targets:
        if target['boxes'].nelement() == 0:
            target['boxes'] = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32, device=device)
            target['area'] = torch.as_tensor([1], dtype=torch.float32, device=device)
            target['labels'] = torch.zeros((1,), dtype=torch.int64, device=device)
            target['iscrowd'] = torch.zeros((1,), dtype=torch.int64, device=device)

    return targets


@torch.no_grad()
def coco_evaluate(data_loader, epoch_outputs, max_dets=None, folder_to_save=None):
    print("Starting COCO mAP eval")
    start = timeit.default_timer()
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types, max_dets=max_dets)

    for i, (_, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = epoch_outputs[i]
        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    if folder_to_save:
        dataset_name = data_loader.dataset.dataset_name
        file_path = os.path.join(folder_to_save, dataset_name + "_coco_map.txt")
        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            print("IoU metric: {}".format(iou_type), file=open(file_path, 'a+'))
            print(coco_eval.stats, file=open(file_path, 'a+'))

    stop = timeit.default_timer()
    total_time = stop - start

    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("COCO mAP Eval ended. Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    return coco_evaluator


def compute_map(dets_and_gts_dict):
    print("Starting mAP Eval")
    start = timeit.default_timer()
    dets_and_gts = []

    for img_id, img_dets_and_gts in dets_and_gts_dict.items():
        img_w, img_h = img_dets_and_gts['img_dim']
        pred_bbs = np.array([[bb[0]/img_w, bb[1]/img_h, bb[2]/img_w, bb[3]/img_h] for bb in img_dets_and_gts['pred_bbs']])
        pred_labels = np.array(img_dets_and_gts['labels'])
        pred_labels = np.zeros(len(pred_labels), dtype=np.int32)
        pred_scores = np.array(img_dets_and_gts['scores'])
        gt_bbs = np.array([[bb[0]/img_w, bb[1]/img_h, bb[2]/img_w, bb[3]/img_h] for bb in img_dets_and_gts['gt_bbs']])
        gt_labels = np.zeros(len(gt_bbs), dtype=np.int32)

        dets_and_gts.append((pred_bbs, pred_labels, pred_scores, gt_bbs, gt_labels))

    mAP = DetectionMAP(1)
    for i, frame in enumerate(dets_and_gts):
        #show_frame(*frame)
        mAP.evaluate(*frame)

    #mAP.plot()
    #plt.show()
    #plt.savefig("./output/training/pr_curve_example.png")

    det_map = mAP.compute_map()

    stop = timeit.default_timer()
    total_time = stop - start

    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("mAP Eval ended. Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    return det_map


def compute_yolo_like_map():
    pass


def compute_dice_and_jaccard(dets_and_gts_dict, smooth=1):
    print("Starting Dice Score and Jaccard Index Computation")
    start = timeit.default_timer()

    imgs_dice, imgs_jaccard = [], []
    for img_id, img_dets_and_gts in dets_and_gts_dict.items():
        img_w, img_h = img_dets_and_gts['img_dim']

        gt_seg_map = np.zeros((img_h, img_w), dtype=np.float32)
        for gt_bb in img_dets_and_gts['gt_bbs']:
            gt_seg_map[int(gt_bb[1]):int(gt_bb[3])+1, int(gt_bb[0]):int(gt_bb[2])+1] = 1.0

        det_seg_map = np.zeros((img_h, img_w), dtype=np.float32)
        for det_bb, score in zip(img_dets_and_gts['pred_bbs'], img_dets_and_gts['scores']):
            seg_map = np.zeros((img_h, img_w), dtype=np.float32)
            seg_map[int(det_bb[1]):int(det_bb[3])+1, int(det_bb[0]):int(det_bb[2])+1] = score
            det_seg_map = np.where(seg_map > det_seg_map, seg_map, det_seg_map)

        intersection = np.sum(gt_seg_map * det_seg_map)
        union = np.sum(gt_seg_map) + np.sum(det_seg_map)
        imgs_dice.append((2 * intersection + smooth) / (union + smooth))

        total = np.sum(gt_seg_map + det_seg_map)
        union = total - intersection
        imgs_jaccard.append((intersection + smooth) / (union + smooth))

    stop = timeit.default_timer()
    total_time = stop - start

    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Dice score and Jaccard coefficient computation ended. Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    return sum(imgs_dice) / len(imgs_dice), sum(imgs_jaccard) / len(imgs_jaccard)





