from tinynn.graph.quantization.quantizer import QATQuantizer
import torch
import os
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import tensorflow as tf
import warnings
from model import BaselineCNN
import sys
import torch.nn.utils.prune as prune
import torch.optim as optim
from timm.utils import model_ema
from dataset import data_loader
from config.config_loader import load_config
from train import epoch_train, evaluation
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')


def quantize(model):
    dummy_input = torch.rand((1, 3, 224, 224))

    quantizer = QATQuantizer(model, dummy_input, work_dir='out')
    q_model = quantizer.quantize()

    for _ in range(10):
        dummy_input = torch.randn(1, 3, 224, 224)
        q_model(dummy_input)

    q_model = torch.quantization.convert(q_model)

    print(q_model)
    print(q_model(dummy_input))

def network_prune(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    module = model.backbone
    module.train()
    pruner = OneShotChannelPruner(module, dummy_input, config={'sparsity': 0.1, 'metrics': 'l2_norm'})

    st_flops = pruner.calc_flops()
    pruner.prune()

    ed_flops = pruner.calc_flops()
    print(f"Pruning over, reduced FLOPS {100 * (st_flops - ed_flops) / st_flops:.2f}%  ({st_flops} -> {ed_flops})")
    
    model.backbone = module
    model.update_features(module(dummy_input).shape[1])
    return model

def weight_prune(model, pruning_perc):
    module = model.backbone
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    with torch.no_grad():
        for p in model.parameters():
            if len(p.data.size()) != 1:

                pruned_inds = p.data.abs() > threshold
                p = p.half() * pruned_inds.int().half()
                p.requires_grad = True
                
                # p = pruned_inds.float()

    model.backbone = module

    return model


def main(
        mode = "weight_prune",
        device= "cuda"
    ):

    config = load_config()

    best_checkpoint_path = config['DIRECTORY']['BEST_MODEL_PATH']
    train_epoch = config["PARAMETER"]["TRAIN_EPOCH"]
    checkpoint_path = config["DIRECTORY"]["OPTIMIZE_MODEL_PATH"]
    batch_size = config["PARAMETER"]["BATCH_SIZE"]
    

    
    train_loader, val_loader = data_loader(batch_size)
    checkpoint = torch.load(best_checkpoint_path)
    model = BaselineCNN('mobilenetv2_050')
    model.load_state_dict(checkpoint["model"])
    model.train()

  
   
    if (mode == "weight_prune"):
        model = weight_prune(model, 50.)
    elif (mode == "network_prune"):
        model = network_prune(model)
    else:
        print("Mode is not available! ")
        exit()
    
    
    ema = model_ema.ModelEmaV2(model, decay=0.9999).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=train_epoch, steps_per_epoch=len(train_loader), three_phase=True)
    if (device == "cuda"):
        model.cuda()
    
    best_metric = 0 
    for epoch in range(train_epoch):
        print(f'\nEpoch {epoch + 1}/{train_epoch}')
        print('-'*30)
        model.train()
        epoch_train(epoch, model, train_loader,
                    scaler, optimizer, scheduler, ema)

        model.eval()
        metric = evaluation(epoch, model, val_loader)

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric
        }
        torch.save(checkpoint, checkpoint_path)

        if (metric > best_metric):
            best_metric = metric
            checkpoint.update({"best_metric": best_metric})
            best_checkpoint_path = checkpoint_path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_checkpoint_path)


if __name__ == "__main__":
    main()
