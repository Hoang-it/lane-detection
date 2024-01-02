import os.path as osp
import numpy as np
import json
from models.clrnet.dataset.base_dataset import BaseDataset
from models.clrnet.utils.tusimple_metric import LaneEval
import random
from torch.utils.data import DataLoader
from mmengine.runner import Runner

SPLIT_FILES = {
    'trainval': dict(anns=[r'train_set\label_data_0313.json', r'train_set\label_data_0601.json', r'train_set\label_data_0531.json'], prefix='train_set'),
    'train': dict(anns=[r'train_set\label_data_0313.json', r'train_set\label_data_0601.json'], prefix='train_set'),
    'val':  dict(anns=[r'train_set\label_data_0531.json'], prefix='train_set'),
    'test': dict(anns=['test_label.json'], prefix='test_set'),
}

class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg:dict=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files, self.prefix = SPLIT_FILES[split].values()
        self.size_limit = cfg.get('size_limit')
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info(f'Loading TuSimple annotations from {self.anno_files}...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines[:self.size_limit]:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                         for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, self.prefix, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, self.prefix, mask_path),
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.get('ori_img_h')
        print(ys)
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.get('ori_img_h')).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir,
                                     'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename,
                                                self.cfg.test_json_file)
        self.logger.info(result)
        return acc
         
def build_tusimple_dataloader(root: str = r'F:\LuanVan\Datasets\TUSimple'):

    # define data loader
    import torch
    import torchvision.transforms as transforms

    norm_cfg = dict(mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg)
    ])

    cfg = dict(
        cut_height = 0,
        img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.]),
        ori_img_w = 1280,
        ori_img_h = 720,
        img_h = 320,
        img_w = 800, 
        num_points = 72,
        max_lanes = 5,
        size_limit = 32000
    )

    from models.clrnet.dataset.process.generate_lane_line import GenerateLaneLine
    import models.clrnet.dataset.process.transforms as  clrtransforms

    img_h = 320
    img_w = 800

    train_processes = [
        GenerateLaneLine(
            cfg=cfg,
            transforms=[
                dict(name='Resize',
                    parameters=dict(size=dict(height=img_h, width=img_w)),
                    p=1.0),
                dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
                dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
                dict(name='MultiplyAndAddToBrightness',
                    parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                    p=0.6),
                dict(name='AddToHueAndSaturation',
                    parameters=dict(value=(-10, 10)),
                    p=0.7),
                dict(name='OneOf',
                    transforms=[
                        dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                        dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                    ],
                    p=0.2),
                dict(name='Affine',
                    parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                            y=(-0.1, 0.1)),
                                    rotate=(-10, 10),
                                    scale=(0.8, 1.2)),
                    p=0.7),
                dict(name='Resize',
                    parameters=dict(size=dict(height=img_h, width=img_w)),
                    p=1.0),
            ]
        ),
        clrtransforms.ToTensor(keys=['img', 'lane_line', 'seg']),
        
    ]

    val_process = [
        GenerateLaneLine(
            cfg=cfg,
            transforms=[
                dict(name='Resize',
                    parameters=dict(size=dict(height=img_h, width=img_w)),
                    p=1.0),
            ],
            training=False),
        clrtransforms.ToTensor(keys=['img']),
    ]

    dataset = TuSimple(
        root,
        split='trainval',
        cfg=cfg,
        processes=train_processes
    )
    val= TuSimple(
        root,
        split='test',
        cfg=cfg,
        processes=val_process
    )
    test= TuSimple(
        root,
        split='test',
        cfg=cfg,
        processes=val_process
    )
        
    print(f"Load dataset with size : {len(dataset)}")
    print(f"Load test with size : {len(test)}")
    print(f"Load val with size : {len(val)}")

    # visualize
    # from utils.visualization import imshow_lanes
    # idx = 1
    # imshow_lanes(dataset[idx]['img'], dataset[idx]['lanes'], show=True)

    batch_size = 16 
    train_dataloader = dict(
        batch_size = batch_size,
        dataset = dataset,
        sampler = dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate')
    )

    val_dataloader = dict(
        batch_size = batch_size,
        dataset = val,
        sampler = dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate')
    )

    test_dataloader = dict(
        batch_size = batch_size,
        dataset = test,
        sampler = dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate')
    )

    train_dataloader = Runner.build_dataloader(train_dataloader)
    val_dataloader = Runner.build_dataloader(val_dataloader)
    test_dataloader = Runner.build_dataloader(test_dataloader)
    
    return train_dataloader, val_dataloader, test_dataloader