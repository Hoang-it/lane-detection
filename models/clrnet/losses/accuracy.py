import mmcv
import torch.nn as nn
import torch
from models.clrnet.losses.lineiou_loss import line_iou
def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    
    # print(f"pred.ndim {pred.ndim}")
    # print(f"target.ndim {target.ndim}")
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh
    
    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
    
    
    def process(self, data_batch, data_samples):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        pred = data_samples[0]
        target = data_samples[1]
        
        # print(pred.shape)
        # print(target.shape)
        # print("=========================")
        # for pre in pred:
        #     if type(pre) is torch.Tensor:
        #         print(pre.shape)
        #         print(pre[0])
        #     else:
        #         print(pre['img'].shape)
        #         print(pre['meta'])
        # print("=========================")
        # for tar in target:
        #     if type(tar) is torch.Tensor:
        #         print(tar.shape)
        #     else:
        #         print(tar['img'].shape)
        #         print(tar['lane-line'].shape)
        #         print(tar['meta'])
        # print("=========================")
        # print(pred[0][0])
        # print(target.shape)
        # print(f"data_batch {len(data_batch)}")
        # print(f"data_batch {data_batch[0].shape}")
        # print(f"data_samples {data_samples[1].shape}")
        # print(f"data_samples {len(data_samples)}")
        score, gt = data_batch
        # save the middle result of a batch to `self.results`
        correct = line_iou(pred, target, 800)
        if len(self.results) == 0 or correct.shape[0] == self.results[-1]['correct'].shape[0]:
            self.results.append({
                'batch_size': len(gt),
                'correct': correct,
            })
    
    def compute_metrics(self, results):
        batch_size = results[0]['batch_size']
        last_size = results[-1]['batch_size']
        
        if last_size != batch_size:
            del results[-1] # need to find a way to add last epoch acc
                
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy= 100 * torch.sum(total_correct) / total_size)
