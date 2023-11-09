from typing import Any, Callable, List
import numpy as np
from mmengine.infer import BaseInferencer
# import mmcv

class CustomInferrencer(BaseInferencer):
    preprocess_kwargs = {'a'}
    forward_kwargs = {'b'}
    visualize_kwargs = {'c'}
    postprocess_kwags = {'d'}
    
    def _init_pipeline(self, cfg) -> Callable[..., Any]:
        pass
    
    def preprocess(self, inputs, batch_size: int = 1, a=None):
        pass
    
    def forward(self, inputs, b=None):
        pass
    
    def visualize(self, inputs: list, preds: Any, show: bool = False, c=None):
        pass
    
    def postprocess(self, preds: Any, visualization: List[np.ndarray], return_datasample=False, d=None) -> dict:
        pass
    
    def __call__(self, 
                 inputs, 
                 return_datasamples: bool = False, 
                 batch_size: int = 1, 
                 a=None,
                 b=None,
                 c=None,
                 d=None,
                 ) -> dict:
        return super().__call__(inputs, return_datasamples, batch_size, a=a, b=b, c=c, d=d)
    
# define inferencer
from mmengine import Config

cfg = Config.fromfile(r'C:\Users\admin\mmengine\work_dir\20231105_113738.py')
weight = r'C:\Users\admin\mmengine\work_dir\epoch_5.pth'

inferencer = CustomInferrencer(model=cfg, weights=weight)

# perform
img = 'F:\LuanVan\Datasets\balloon\train\126700562_8e27720147_b.jpg'
result = inferencer(img)

# mmcv.imshow(result)