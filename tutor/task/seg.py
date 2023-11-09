import os
from typing import Optional, Sequence
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
import csv

# define dataset
def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
        
    return color_to_class

class CamVid(VisionDataset):
    def __init__(self,
                 root,
                 img_folder,
                 mask_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)
        
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.images = list(sorted(os.listdir(os.path.join(self.root, img_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, self.mask_folder))))
        self.color_to_class = create_palette(os.path.join(self.root, 'class_dict.csv'))
        
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder, self.masks[index])
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert RGB value to it class
        mask = np.array(mask)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        labels = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            labels[mask == rgb] = class_index
            
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        
        data_samples = dict(labels=labels,
                            img_path=img_path,
                            mask_path=mask_path)
        return img, data_samples
    
    def __len__(self):
        return len(self.images)            
        
# define data loader
import torch
import torchvision.transforms as transforms

norm_cfg = dict(mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**norm_cfg)
])

target_transform = transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))

train_set = CamVid(
    'data\CamVid',
    img_folder='train',
    mask_folder='train_labels',
    transform=transform,
    target_transform=target_transform
)

val_set = CamVid(
    'data\CamVid',
    img_folder='val',
    mask_folder='val_labels',
    transform=transform,
    target_transform=target_transform
)

train_dataloader = dict(
    batch_size = 3,
    dataset = train_set,
    sampler = dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

val_dataloader = dict(
    batch_size = 3,
    dataset = val_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn = dict(type='default_collate')
)

# Model

from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F

class MMDeepLabV3(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = deeplabv3_resnet50(num_classes=num_classes)

    def forward(self, imgs, data_samples=None, mode='tensor'):
        x = self.deeplab(imgs)['out']
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples['labels'])}
        elif mode == 'predict':
            return x, data_samples
        
# define metric
from mmengine.evaluator import BaseMetric

class IoU(BaseMetric):
    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds = torch.argmax(preds, dim=1)
        intersect = (labels == preds).sum()
        union = (torch.logical_or(preds, labels)).sum()
        iou = (intersect/union).cpu()
        self.results.append(dict(
            batch_size=len(labels),
            iou = iou*len(labels)
        ))
    
    def compute_metrics(self, results: list) -> dict:
        total_iou = sum(result['iou'] for result in results)
        num_samples = sum(result['batch_size'] for result in results)
        return dict(iou = total_iou / num_samples)

# Custom hook
from mmengine.hooks import Hook
import shutil
import cv2
import os.path as osp

class SegVisHook(Hook):
    def __init__(self, data_root, vis_num=1):
        super().__init__()
        self.vis_num = vis_num
        self.palette = create_palette(osp.join(data_root, 'class_dict.csv'))
    
    def after_val_iter(self, 
                       runner, 
                       batch_idx: int, 
                       data_batch = None, 
                       outputs= None) -> None:
        if batch_idx > self.vis_num:
            return
        
        preds, data_samples = outputs
        img_paths = data_samples['img_path']    
        mask_paths = data_samples['mask_path']   
        _, C, H, W = preds.shape 
        
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            runner.visualizer.set_image(pred_mask)
            for color, class_id in self.palette.items():
                runner.visualizer.draw_binary_masks(
                    pred == class_id,
                    colors=[color],
                    alphas=1.0,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            saved_dir = osp.join(runner.log_dir, 'vis_data', str(idx))
            os.makedirs(saved_dir, exist_ok=True)

            shutil.copyfile(img_path,
                            osp.join(saved_dir, osp.basename(img_path)))
            shutil.copyfile(mask_path,
                            osp.join(saved_dir, osp.basename(mask_path)))
            cv2.imwrite(
                osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
                pred_mask)

# define runnerhoan
from torch.optim import AdamW 
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner

num_classes = 32
runner = Runner(
    model=MMDeepLabV3(num_classes=num_classes),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(
        type=AmpOptimWrapper, # this help training more faster
        optimizer=dict(type=AdamW, lr=2e-4)
    ),
    train_cfg=dict(by_epoch=True, max_epochs = 10, val_interval = 10),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=IoU),
    custom_hooks=[SegVisHook('data/CamVid')],
    default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1)),  
    cfg=dict(model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)),
    # for resume  
    load_from='./work_dir/epoch_1.pth',
    resume=True,
)

from mmengine.analysis import get_model_complexity_info

input_shape = (3, 224, 224)
# model= MMDeepLabV3(4)
# analysis_results = get_model_complexity_info(model, input_shape)

# run
if __name__ == '__main__':
    # print('Model Flops:{}'.format(analysis_results['flops_str']))
    # print('Model Parameters:{}'.format(analysis_results['params_str']))
    # print(analysis_results.keys())

    runner.train()