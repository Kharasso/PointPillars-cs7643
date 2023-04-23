import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from compas import geometry
from utils import process
class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1/9, cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.dir_cls = nn.CrossEntropyLoss()
    
    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels, 
                num_cls_pos, 
                batched_bbox_reg, 
                batched_dir_labels):
        '''
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float() # (n, 3)

        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
             (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels) # (n, 3)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos
        
        # 2. regression loss
        # reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        # reg_loss = reg_loss.sum() / reg_loss.size(0)
        '''
        pred_points = self.points_from_bbox(bbox_pred)  # (n,8,3)
        gt_points = self.points_from_bbox(batched_bbox_reg)
        # points = torch.cat((pred_points, gt_points), dim=0).cpu().detach().numpy() #(2n,8,3)
        bbox_diag = []
        for cnt in range(pred_points.size()[0]):
            points = torch.cat((pred_points[cnt, :, :], gt_points[cnt, :, :]), dim=1).squeeze(dim=0).cpu().detach().numpy()
            min_bbox = geometry.oriented_bounding_box_numpy(points)
        #min_bbox_w = geometry.length_vector(geometry.subtract_vectors(min_bbox[1], min_bbox[0]))
        #min_bbox_l = geometry.length_vector(geometry.subtract_vectors(min_bbox[3], min_bbox[0]))
        #min_bbox_h = geometry.length_vector(geometry.subtract_vectors(min_bbox[4], min_bbox[0]))
            min_bbox_diag = geometry.length_vector(geometry.subtract_vectors(min_bbox[6], min_bbox[0]))
            bbox_diag.append(min_bbox_diag)
        bbox_diag = torch.tensor(bbox_diag).to(bbox_pred.device)
        iou = torch.diag(process.iou3d(bbox_pred, batched_bbox_reg), 0)  # (n,1)
        l2_distance = torch.diag(torch.cdist(bbox_pred[:, 0:3], batched_bbox_reg[:, 0:3]), 0)  # (n,1)

        reg_loss = (1 - iou + torch.square_(l2_distance)/torch.square_(bbox_diag))  # L_DIOU
        reg_loss = reg_loss.sum()/reg_loss.size(0)
        '''

        # reg_loss = self.lossEIOU(bbox_pred, batched_bbox_reg)
        #print(reg_loss.size())  # (n,1)
        # reg_loss = reg_loss.sum()/reg_loss.size(0)

        reg_loss = self.mse_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)
        # 3. direction cls loss
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)

        # 4. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        
        loss_dict={'cls_loss': cls_loss, 
                   'reg_loss': reg_loss,
                   'dir_cls_loss': dir_cls_loss,
                   'total_loss': total_loss}
        return loss_dict

    def points_from_bbox(self, bbox):
        '''
        bbox_pred: (n, 7) (x,y,z,w,l,h,theta)
        return: points, (n, 8, 3). 8points to construct the bbox
        '''

        p0 = torch.cat(((bbox[:, 0] + 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] + 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] + 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                        dim=1).unsqueeze(dim=1)  # (n,1,3)
        p1 = torch.cat(((bbox[:, 0] + 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] + 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] - 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                        dim=1).unsqueeze(dim=1)  # (n,1,3)
        p2 = torch.cat(((bbox[:, 0] + 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] - 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] + 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)
        p3 = torch.cat(((bbox[:, 0] + 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] - 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] - 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)
        p4 = torch.cat(((bbox[:, 0] - 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] + 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] + 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)
        p5 = torch.cat(((bbox[:, 0] - 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] + 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] - 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)
        p6 = torch.cat(((bbox[:, 0] - 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] - 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] + 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)
        p7 = torch.cat(((bbox[:, 0] - 0.5 * bbox[:, 3]).unsqueeze(dim=1), \
                        (bbox[:, 1] - 0.5 * bbox[:, 4]).unsqueeze(dim=1), \
                        (bbox[:, 2] - 0.5 * bbox[:, 5]).unsqueeze(dim=1)),\
                          dim=1).unsqueeze(dim=1)  # (n,1,3)

        final_p = torch.cat((p0, p1, p2, p3, p4, p5, p6, p7), dim=1)  # (n,8,3)
        #bbox1 = bbox[:, 0] - 0.5 * bbox[:, 3]
        #bbox2 = bbox[:, 1] + 0.5 * bbox[:, 4]
        ##bbox3 = bbox[:, 1] - 0.5 * bbox[:, 4]
        #bbox4 = bbox[:, 2] + 0.5 * bbox[:, 5]
        #bbox6 = bbox[:, 2] - 0.5 * bbox[:, 5]
        #l = []
        #l.append(b)
        return final_p
    def lossEIOU(self, bbox, bboxgt):
        '''
        bbox: (n, 7) (x,y,z,w,l,h,theta) bbox predit
        bboxgt: (n, 7) box ground true
        return: loss, (n, 1). 8points to construct the bbox
        '''
        pl = bbox[:, 0] - 0.5 * bbox[:, 3]
        pr = bbox[:, 0] + 0.5 * bbox[:, 3]
        pt = bbox[:, 1] + 0.5 * bbox[:, 4]
        pb = bbox[:, 1] - 0.5 * bbox[:, 4]
        pfw = bbox[:, 2] + 0.5 * bbox[:, 5]
        pbw = bbox[:, 2] - 0.5 * bbox[:, 5]

        gl = bboxgt[:, 0] - 0.5 * bboxgt[:, 3]
        gr = bboxgt[:, 0] + 0.5 * bboxgt[:, 3]
        gt = bboxgt[:, 1] + 0.5 * bboxgt[:, 4]
        gb = bboxgt[:, 1] - 0.5 * bboxgt[:, 4]
        gfw = bboxgt[:, 2] + 0.5 * bboxgt[:, 5]
        gbw = bboxgt[:, 2] - 0.5 * bboxgt[:, 5]

        w_intersect = torch.min(pl, gl) + torch.min(pr, gr)
        g_w_intersect = torch.min(pl, gl) + torch.max(pr, gr)

        l_intersect = torch.min(pt, gt) + torch.min(pb, gb)
        g_l_intersect = torch.max(pt, gt) + torch.min(pb, gb)

        h_intersect = torch.min(pfw, gfw) + torch.min(pbw, gbw)
        g_h_intersect = torch.max(pfw, gfw) + torch.min(pbw, gbw)

        w_union = g_w_intersect + 1e-7
        l_union = g_l_intersect + 1e-7
        h_union = g_h_intersect + 1e-7

        loss_w = ((bbox[:, 3] - bboxgt[:, 3])**2) / (w_union ** 2)
        loss_l = ((bbox[:, 4] - bboxgt[:, 4]) ** 2) / (l_union ** 2)
        loss_h = ((bbox[:, 5] - bboxgt[:, 5]) ** 2) / (h_union ** 2)

        sim_wlh = (loss_w + loss_l + loss_h) / 3

        l2_distance = torch.diag(torch.cdist(bbox[:, 0:3], bboxgt[:, 0:3]), 0)  # (n,1)
        bounding_box_diag_distance = w_union ** 2 + l_union ** 2 + h_union ** 2 + 1e-7
        sim_xyz = l2_distance / bounding_box_diag_distance

        iou = torch.diag(process.iou3d(bbox, bboxgt), 0)

        eiou = 1 - iou + (1 - iou) * (sim_wlh + sim_xyz)

        return eiou


