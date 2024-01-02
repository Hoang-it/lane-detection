import os

# define dataset                 
from config.clrnet.tusimple.load_dataloader_tusimple import train_dataloader, test_dataloader, val_dataloader

# Model
from config.clrnet.tusimple.load_model_tusimple import model

# Loss
from models.clrnet.losses.accuracy import Accuracy

# define runner
from torch.optim import AdamW 
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner

runner = Runner(
    model=model,
    work_dir='./work_di/tusimple',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(
        type=AmpOptimWrapper, # this help training more faster
        optimizer=dict(type=AdamW, lr=1.0e-3)
    ),
    train_cfg=dict(by_epoch=True, max_epochs = 70, val_interval = 10),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    test_dataloader=test_dataloader,
    test_cfg=dict(),
    test_evaluator=dict(type=Accuracy),
    default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1)),  
    cfg=dict(model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)),
    # for resume  
    load_from='./work_di/tusimple/epoch_24.pth',
    # resume=True,
)

# run
if __name__ == '__main__':
    runner.train()



