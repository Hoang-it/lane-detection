from typing import Any, Optional, Sequence, Union
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

# define model
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        
    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
        
# define dataset and dataloader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from clrnet.load_dataloader_tusimple import dataset, val

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=2,
                              shuffle=True,
                              dataset=dataset)

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=val)

# define evaluation metrics
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        
        return dict(accuracy= 100* total_correct/total_size)

# define runner
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy)
)

# run train

runner.train()