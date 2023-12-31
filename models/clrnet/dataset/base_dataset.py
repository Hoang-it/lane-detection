import os.path as osp
import cv2
from torchvision.datasets import VisionDataset
import logging
from models.clrnet.utils.visualization import imshow_lanes
from models.clrnet.dataset.process import Process
    
class BaseDataset(VisionDataset):
    def __init__(self, data_root, split, processes=None, cfg:dict=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)
        self.data_infos = []
        
    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info: dict = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        if img is not None:
            img = img[self.cfg.get('cut_height'):, :, :]
            sample = data_info.copy()
            sample.update({'img': img})

            if self.training:
                label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
                if len(label.shape) > 2:
                    label = label[:, :, 0]
                label = label.squeeze()
                label = label[self.cfg.get('cut_height'):, :]
                sample.update({'mask': label})

                if self.cfg.get('cut_height') != 0:
                    new_lanes = []
                    for i in sample['lanes']:
                        lanes = []
                        for p in i:
                            lanes.append((p[0], p[1] - self.cfg.get('cut_height')))
                        new_lanes.append(lanes)
                    sample.update({'lanes': new_lanes})

            sample = self.processes(sample)
            meta = {'full_img_path': data_info['img_path'],
                    'img_name': data_info['img_name']}
            # meta = BaseDataElement(meta, cpu_only=True)
            sample.update({'meta': meta})

            return img, sample
        else:
            print(f'Can not found img from path : {data_info["img_path"]}')            
