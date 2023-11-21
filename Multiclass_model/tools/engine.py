import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import copy
from timm.utils import model_ema
from data.prefetch import PrefetchLoader




def get_metric(
        current_label: int,
        f1_pred: np.array, 
        f1_label: np.array, 
        thres: float = 0.5
    ):
    
    f1_pred[f1_pred > thres] = 1
    f1_pred[f1_pred <= thres] = 0
    precision, recall, f_score, _ = precision_recall_fscore_support(f1_label[current_label],f1_pred[current_label],average='micro')

    return precision,recall,f_score

def update_metric(
    input,
    new_metric: np.array
):
    if (isinstance(input,torch.Tensor)):
        input = torch.transpose(input,0,1).cpu().detach().numpy()
    else:
        input = np.transpose(input)
    new_metric = np.concatenate((new_metric,input),axis=1)
    
    return new_metric

def epoch_train(
        epoch: int, 
        model: torch.nn.modules,
        dataloader: PrefetchLoader,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        optimizer: torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler.OneCycleLR,
        ema: model_ema.ModelEmaV2,
        writer = None,
        config: dict = None
    ):
    
    device = config['PARAMETER']['DEVICE']
    running_loss = 0
    
    for (input, labels) in dataloader:
        inputs = input.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        labels = torch.transpose(labels,0,1)
        outputs = model(inputs,labels,'Training')
        
        loss = sum(outputs[idx] for idx in outputs.keys())
        losses = loss 

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
           
        ema.update(model)
        
        running_loss += loss.item()
    
    
    epoch_loss = running_loss/(len(dataloader))
    if (writer != None):
        writer.add_scalar('training loss mobile_2',
                                epoch_loss,
                                epoch)

    
    print(f'Epoch loss: {epoch_loss:.4f} \n')
    
    
def evaluation(
        epoch: int, 
        model: torch.nn.modules,
        dataloader: PrefetchLoader,
        writer = None,
        config: dict = None
    ):
    
    keyword = config['LABEL_OPTION']['LABEL']
    NUM_OF_LABEL = config['LABEL_OPTION']['NUM_OF_LABEL']
    device = config['PARAMETER']['DEVICE']
    
    f1_pred = [[] for i in range(NUM_OF_LABEL)]
    f1_label = [[] for i in range(NUM_OF_LABEL)]
    
    for (input, labels) in dataloader:
        inputs = input.to(device)
        labels = labels.to(device)
        # print(labels)
        outputs = model(inputs,mode='Validation')
    
        f1_pred = update_metric(outputs, f1_pred)
        f1_label = update_metric(labels, f1_label)
        

    
    metric = 0
    for current_label in range(len(keyword)): 
        precision, recall, f_score = get_metric(current_label,f1_pred,f1_label)
        print(f'Result of {keyword[current_label]}: ')
        print(f'F-score: {f_score:.4f} Precision: {precision:.4f}, Recall: {recall:.4f} \n')
        metric += f_score
        
    metric /= len(keyword)           
    print(f'Average F1 score: {metric:.4f}')
  
    return metric
        