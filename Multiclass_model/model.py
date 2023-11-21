import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torch.cuda.amp import autocast
from torchvision.transforms.functional import normalize
from functools import partial

def build_ml_heads(num_features):
    in_channels = num_features
    out_channels = 2
    s = 32
    m = 0.35
    num_centers = 1
    sub_center_type = "max"
    head = ArcFace(in_channels, out_channels, s, m, num_centers, sub_center_type)
    return head


class ArcFace(nn.Module):
    """
    This module implements ArcFace.
    """

    def __init__(
        self, in_channels, out_channels, s=32.0, m=0.5, num_centers=1, sub_center_type="max"
    ):
        super(ArcFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.m = m
        self.num_centers = num_centers
        self.sub_center = num_centers > 1
        self.sub_center_type = sub_center_type
        self.training  = True
        self.weight = nn.Parameter(torch.Tensor(out_channels * num_centers, in_channels))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ", num_centers="
            + str(self.num_centers)
            + ", sub_center_type="
            + self.sub_center_type
            + ")"
        )

    def forward(self, inputs, labels):
        # cos(theta)
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            if self.sub_center_type == "softmax":
                cosine = F.softmax(cosine * self.s, 1) * cosine
                cosine = cosine.sum(1)
            elif self.sub_center_type == "max":
                cosine = cosine.max(1)[0]
        if not self.training or labels is None:
            return {"pred_class_logits": cosine * self.s}
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return {"cls_outputs": output, "pred_class_logits": cosine * self.s}


def build_timm_backbone(config):
    kwargs = {}
    kwargs["model_name"] = config['BACKBONE']
    kwargs["pretrained"] = config['PRETRAINED']
    kwargs["drop_path_rate"] = config['DROP_PATH_RATE']
    kwargs["num_classes"] = 0
    if kwargs["model_name"] == "resnest269e":
        # drop path is not supported
        kwargs.pop("drop_path_rate")
    out_indices = {}

    if len(out_indices) > 0:
        kwargs["out_indices"] = out_indices
        model = create_model(**kwargs)
    else:
        model = create_model(**kwargs)
        # model.reset_classifier(0, "")

    return model

def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    """
    Functional version of CrossEntropyLoss with label/adaptive label smoothing.
    """
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else: 
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(
            1
        )

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1).long(), (1 - smooth_param))
    loss = (-targets * log_probs).sum(dim=1)
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    loss = loss.sum() / non_zero_cnt
    return loss



class BaselineCNN(nn.Module):
    """
    Baseline models.
    """

    def __init__(self, model_name):
        super(BaselineCNN, self).__init__()
        
        self.backbone = build_timm_backbone(model_name)
        num_features = self.backbone.num_features
        
        self.banh_chung = build_ml_heads(num_features)
        self.cay_dao = build_ml_heads(num_features)
        self.cay_mai = build_ml_heads(num_features)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.m = nn.Softmax(dim=1)
        self.loss_fn = partial(
            cross_entropy_loss,
            eps=0,
            alpha=0.2,
        )
    def _forward_features(self, images):
        images = normalize(
            images,
            mean=self.mean,
            std=self.std,
        )
        features = self.backbone(images)

        return features

    def update_features(self, num_features): 
        self.banh_chung = build_ml_heads(num_features)
        self.cay_dao = build_ml_heads(num_features)
        self.cay_mai = build_ml_heads(num_features)
        
    @autocast()
    def forward(self, images, labels = None, mode = 'None'):
        features = self._forward_features(images)

        if (labels == None):
            banhchung = self.banh_chung(features,None),
            caydao = self.cay_dao(features,None),
            caymai = self.cay_mai(features,None),
        else:
            banhchung = self.banh_chung(features,labels[0]),
            caydao = self.cay_dao(features,labels[1]),
            caymai = self.cay_mai(features,labels[2]),

        # print(banhchung[0]['pred_class_logits'])
        # print(labels[0])
        if (mode == 'Training'):
            return {
                'banh_chung':self.loss_fn(banhchung[0]['cls_outputs'],labels[0]),
                'cay_dao':self.loss_fn(caydao[0]['cls_outputs'],labels[1]),
                'cay_mai':self.loss_fn(caymai[0]['cls_outputs'],labels[2]),
            }
        elif (mode == 'Validation'):
            # print(banh_chung)
            # print(self.m(banhchung['pred_class_logits']))
            # print(torch.argmax(self.m(banhchung['pred_class_logits']),1))
            # print( torch.unsqueeze(torch.argmax(self.m(banhchung[0]['pred_class_logits']),1),1))
            banh_chung = torch.max(self.m(banhchung[0]['pred_class_logits']),1)
            cay_dao = torch.max(self.m(caydao[0]['pred_class_logits']),1)
            cay_mai = torch.max(self.m(caymai[0]['pred_class_logits']),1)
            banh_chung_idx = torch.argmax(self.m(banhchung[0]['pred_class_logits']),1)
            cay_dao_idx = torch.argmax(self.m(caydao[0]['pred_class_logits']),1)
            cay_mai_idx = torch.argmax(self.m(caymai[0]['pred_class_logits']),1)
            for i in range(len(banh_chung[0])):
                banh_chung[0][i] = 1 - abs(banh_chung_idx[i] - banh_chung[0][i])
                cay_dao[0][i] = 1 - abs(cay_dao_idx[i] - cay_dao[0][i])
                cay_mai[0][i] = 1 - abs(cay_mai_idx[i] - cay_mai[0][i])
            
            
            banh_chung = torch.unsqueeze(banh_chung[0],1)
            cay_dao = torch.unsqueeze(cay_dao[0],1)
            cay_mai = torch.unsqueeze(cay_mai[0],1)
            
            
            

            # banh_chung = torch.unsqueeze(torch.argmax(self.m(banhchung[0]['pred_class_logits']),1),1)
            # cay_dao = torch.unsqueeze(torch.argmax(self.m(caydao[0]['pred_class_logits']),1),1)
            # cay_mai = torch.unsqueeze(torch.argmax(self.m(caymai[0]['pred_class_logits']),1),1)
            logits = torch.cat((banh_chung,cay_dao,cay_mai), 1)
            logits = logits.type(torch.float32)
    
            return logits
        else:
            banh_chung = self.m(banhchung[0]['pred_class_logits'])
            cay_dao = self.m(caydao[0]['pred_class_logits'])
            cay_mai = self.m(caymai[0]['pred_class_logits'])

            logits = torch.cat((banh_chung,cay_dao,cay_mai), 1)
            logits = logits.type(torch.float32)

            return logits
