import os

# define dataset                 
from config.clrnet.tusimple.load_dataloader_tusimple import build_tusimple_dataloader

# Model
from config.clrnet.tusimple.load_model_tusimple import model

# Loss
from models.clrnet.losses.accuracy import Accuracy

# define runner
from torch.optim import AdamW 
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
import sys
import argparse


# run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--load_from', default=None, help='the checkpoint file to load from')
    parser.add_argument('--dataset_root', default=None, help='folder to load dataset')
    parser.add_argument('--size_limit', default=None, help='limit image to load from dataset')
    parser.add_argument('--batch_size', default=None, help='batch size for each epoch')
    args = parser.parse_args()
    
    if args.dataset_root is None:
        print("Load default params")
        train_dataloader, test_dataloader, val_dataloader = build_tusimple_dataloader(size_limit=32)
    elif args.batch_size is None:
        print("Load with dataset root")
        train_dataloader, test_dataloader, val_dataloader = build_tusimple_dataloader(root=args.dataset_root)
    else:
        print("Load with custom param")
        train_dataloader, test_dataloader, val_dataloader = build_tusimple_dataloader(root=args.dataset_root, batch_size=int(args.batch_size), size_limit=320000)
        
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
        load_from= args.load_from, #'./work_di/tusimple/epoch_1.pth'
        # resume=True,
    )
    runner.train()



