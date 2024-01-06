from mmengine.model import BaseModel
from models.clrnet.losses.focal_loss import FocalLoss
import torch
from models.clrnet.utils.dynamic_assign import assign
from models.clrnet.losses.lineiou_loss import liou_loss
import torch.nn.functional as F
from models.clrnet.losses.accuracy import accuracy
from models.clrnet.heads.clr_head import CLRHead

class Detector(BaseModel):
    def __init__(self,
                backbone,
                neck,
                heads,
                cfg: dict,
                aggregator=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.aggregator = aggregator
        self.neck = neck
        self.heads : CLRHead = heads
        self.refine_layers = cfg.get('refine_layers')
        self.img_w = cfg.get('img_w')
        self.img_h = cfg.get('img_h')
        self.n_strips = cfg.get('num_points') - 1
        
        weights = torch.ones(self.cfg.get('num_classes'))
        weights[0] = self.cfg.get('bg_weight')
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.get('ignore_label'), weight=weights)
    
    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        if 'cls_loss_weight' in self.cfg:
            cls_loss_weight = self.cfg.get('cls_loss_weight')
        if 'xyt_loss_weight' in self.cfg:
            xyt_loss_weight = self.cfg.get('xyt_loss_weight')
        if 'iou_loss_weight' in self.cfg:
            iou_loss_weight = self.cfg.get('iou_loss_weight')
        if 'seg_loss_weight' in self.cfg:
            seg_loss_weight = self.cfg.get('seg_loss_weight')

        predictions_lists = output['predictions_lists']
        if isinstance(batch, dict): 
            lane_line : torch.Tensor = batch['lane_line']
            targets = lane_line.clone()
            cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
            cls_loss = 0
            reg_xytl_loss = 0
            iou_loss = 0
            cls_acc = []

            cls_acc_stage = []
            for stage in range(self.refine_layers):
                predictions_list : list[torch.Tensor] = predictions_lists[stage]
                for predictions, target in zip(predictions_list, targets):                    
                    target = target[target[:, 1] == 1]

                    if len(target) == 0:
                        # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                        cls_target = predictions.new_zeros(predictions.shape[0]).long()
                        cls_pred = predictions[:, :2]
                        cls_criterion_loss : torch.Tensor = cls_criterion(cls_pred, cls_target)
                        cls_loss = cls_loss + cls_criterion_loss.sum()
                        continue

                    with torch.no_grad():
                        matched_row_inds, matched_col_inds = assign(predictions, target, self.img_w, self.img_h)

                    # classification targets
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_target[matched_row_inds] = 1
                    cls_pred = predictions[:, :2]

                    # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                    reg_yxtl = predictions[matched_row_inds, 2:6]
                    reg_yxtl[:, 0] *= self.n_strips
                    reg_yxtl[:, 1] *= (self.img_w - 1)
                    reg_yxtl[:, 2] *= 180
                    reg_yxtl[:, 3] *= self.n_strips

                    target_yxtl = target[matched_col_inds, 2:6].clone()

                    # regression targets -> S coordinates (all transformed to absolute values)
                    reg_pred = predictions[matched_row_inds, 6:]
                    reg_pred *= (self.img_w - 1)
                    reg_targets = target[matched_col_inds, 6:].clone()

                    with torch.no_grad():
                        predictions_starts = torch.clamp(
                            (predictions[matched_row_inds, 2] *
                            self.n_strips).round().long(), 0,
                            self.n_strips)  # ensure the predictions starts is valid
                        target_starts = (target[matched_col_inds, 2] *
                                        self.n_strips).round().long()
                        target_yxtl[:, -1] -= (predictions_starts - target_starts
                                            )  # reg length

                    # Loss calculation
                    cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                    ) / target.shape[0]

                    target_yxtl[:, 0] *= self.n_strips
                    target_yxtl[:, 2] *= 180
                    reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                        reg_yxtl, target_yxtl,
                        reduction='none').mean()

                    iou_loss = iou_loss + liou_loss(
                        reg_pred, reg_targets,
                        self.img_w, length=15)

                    # calculate acc
                    cls_accuracy = accuracy(cls_pred, cls_target)
                    cls_acc_stage.append(cls_accuracy)

                cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

            # extra segmentation loss
            seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
                                batch['seg'].long())

            cls_loss /= (len(targets) * self.refine_layers)
            reg_xytl_loss /= (len(targets) * self.refine_layers)
            iou_loss /= (len(targets) * self.refine_layers)

            loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
                + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

            return_value = {
                'loss': loss,
                'loss_stats': {
                    'loss': loss,
                    'cls_loss': cls_loss * cls_loss_weight,
                    'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                    'seg_loss': seg_loss * seg_loss_weight,
                    'iou_loss': iou_loss * iou_loss_weight
                }
            }

            for i in range(self.refine_layers):
                return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

            return return_value
        
    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch, data_samples=None, mode='tensor'):
        
        output = {}
        batch = batch.float()
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        # print(f"fea {fea}")
        if self.training:
            output = self.heads(fea, batch=batch)
            # print(f"out shape: {output}")
        else:
            output = self.heads(fea, batch=batch)

        if mode == 'loss':
            # print(f"out loss shape: {type(output)}")
            # print(f"data_samples shape: {data_samples}")
            # print(f"Output {data_samples['lane_line']}")
            loss = self.loss(output, data_samples)
            # print(f"loss {loss}")
            return dict(loss=loss.get('loss'))
        elif mode == 'predict':
            most_related = output[:, :5, :] #need to caculate how to get 5 lanes most relative
            return most_related, data_samples['lane_line']
        return output