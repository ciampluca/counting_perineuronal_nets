import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN as FasterRCNNTorch, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


MODEL_URLS = {
    # 'fasterrcnn_resnet50_fpn_coco'
    'resnet50': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    # 'fasterrcnn_resnet101_fpn_coco',   Note: NOT official
    'resnet101': 'https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth',
}


class FasterRCNNWrapper(FasterRCNNTorch):
    """
    FasterRCNN
    Ref. S. Ren et al. 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks'
    """
    
    def __init__(self,
        in_channels=3,
        num_classes=1,
        backbone='resnet50',
        backbone_pretrained=False,
        model_pretrained=False,
        max_dets_per_image=200,
        nms=0.3,
        det_thresh=0.05,
        cache_folder='./model_zoo',
        skip_weights_loading=False,
        progress=True,
    ):
    
        assert backbone in ("resnet50", "resnet101"), f"Backbone not supported: {backbone}"

        if skip_weights_loading:
            model_pretrained = False
            backbone_pretrained = False

        # anchor generator: these are default values, but in this way we can eventually change them
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # box roi pooler: these are default values, but in this way we can eventually change them
        box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # defining the backbone (there's no need to download the backbone if model_pretrained is set)
        backbone_pretrained = backbone_pretrained and not model_pretrained
        backbone_module = resnet_fpn_backbone(backbone, backbone_pretrained)

        super().__init__(
            backbone_module,
            num_classes=91,  # for loading COCO pretraining
            box_detections_per_img=max_dets_per_image,
            box_nms_thresh=nms,
            box_score_thresh=det_thresh,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=box_roi_pooler
        )

        if model_pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS[backbone], progress=progress, model_dir=cache_folder)
            self.load_state_dict(state_dict)

        # get number of input features for the classifier
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one that has num_classes which is user-defined
        num_classes += 1    # num classes + background
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)