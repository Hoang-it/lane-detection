from models.clrnet.model.detector import Detector
from models.clrnet.backbone.resnet import ResNetWrapper
from models.clrnet.heads.clr_head import CLRHead
from models.clrnet.necks.fpn import FPN

backbone = ResNetWrapper(
    resnet='resnet18',
    pretrained=False,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

max_lanes = 5
heads = CLRHead(
    num_priors=192,
    refine_layers=3,
    fc_hidden_dim=64,
    sample_points=36,
    cfg=dict(
        img_h = 320,
        img_w = 800,
        num_classes= 6 + 1,
        bg_weight=0.4,
        ignore_label=255,
        max_lanes = max_lanes,
        test_parameters = dict(conf_threshold=0.40, nms_thres=50, nms_topk=max_lanes),
        ori_img_w = 1280,
        ori_img_h = 720,
        cut_height = 160 
    )
)

neck = FPN(
    in_channels=[128, 256, 512],
    out_channels=64,
    num_outs=3,
    attention=False
)
cfg = dict(
    refine_layers=3,
    img_h = 320,
    img_w = 800,
    n_strips = 3,
    num_points = 72,
    num_classes = 6 + 1,
    ignore_label = 255,
    bg_weight = 0.4
)

model = Detector(
    backbone=backbone,
    heads=heads,
    neck=neck,
    cfg=cfg,
)

from mmengine.analysis import get_model_complexity_info

input_shape = (3, 720, 1280)
analysis_results = get_model_complexity_info(model, input_shape)
print(analysis_results['out_table'])