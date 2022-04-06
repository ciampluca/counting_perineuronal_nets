import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.fcos import FCOS as FCOSTorch, FCOSClassificationHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models import resnet
from torchvision.ops import misc, feature_pyramid_network



MODEL_URLS = {
    # 'fcos_resnet50_fpn_coco'
    "resnet50": "https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth",
}


class FCOSWrapper(FCOSTorch):
    
    def __init__(self,
        in_channels=3,
        out_channels=1,
        backbone='resnet50',
        backbone_pretrained=False,
        model_pretrained=False,
        max_dets_per_image=200,
        nms=0.3,
        det_thresh=0.05,
        # TODO dont know exatcly what is it
        center_sampling_radius=1.5,
        cache_folder='./model_zoo',
        skip_weights_loading=False,
        progress=True,
    ):
        
        assert backbone in ("resnet50"), f"Backbone not supported: {backbone}"
            
        if skip_weights_loading:
            model_pretrained = False
            backbone_pretrained = False
            
        # defining the backbone (there's no need to download the backbone if model_pretrained is set)
        backbone_pretrained = backbone_pretrained and not model_pretrained
        
        backbone_module = resnet.__dict__[backbone](pretrained=backbone_pretrained, norm_layer=misc.FrozenBatchNorm2d)
        
        backbone_module = _resnet_fpn_extractor(
            backbone_module, 3, returned_layers=[2, 3, 4], extra_blocks=feature_pyramid_network.LastLevelP6P7(256, 256)
        )       
        
        super().__init__(
            backbone_module,
            num_classes=91,  # for loading COCO pretraining
            detections_per_img=max_dets_per_image,
            score_thresh=det_thresh,
            nms_thresh=nms,
            center_sampling_radius=center_sampling_radius,
        )
        
        if model_pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS[backbone], progress=progress, model_dir=cache_folder)
            self.load_state_dict(state_dict)
            
        # get number of input features for the classifier
        in_features = self.head.classification_head.cls_logits.in_channels
        # get number of anchors per location
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        # replace the pre-trained head with a new one that has num_classes which is user-defined
        out_channels += 1    # num classes + background
        # replace the pre-trained head with a new one
        self.head.classification_head = FCOSClassificationHead(in_features, num_anchors, out_channels)
