import torch
from torch import nn
from torchvision import models

import random
from model import BaselineCNN
from data.dataset import data_loader
from torch.optim import lr_scheduler
import torch.optim as optim
from timm.utils import model_ema
from tools.engine import epoch_train, evaluation
from torch.utils.tensorboard import SummaryWriter
from config.config_loader import load_config
import os


def main():
    """
    Main function to train and evaluate a model.
    """
    # Load configuration
    config = load_config()  
    
    # Extract required parameters from the configuration
    train_epoch = config["TRAIN_OPTION"]["NUM_EPOCH"]
    model_path = config["DIRECTORY"]["MODEL_PATH"]
    backbone = config["BACKBONE_OPTION"]
    writer_path = config["WRITER_OPTION"]["LOG_PATH"]
    
    # Remove existing TensorBoard writer path if it exists
    if (os.path.exists(writer_path)):
        os.system('rm -r ' + writer_path)
    
    # Set device
    device = torch.device(config["PARAMETER"]["DEVICE"])

    # Initialize model
    model = BaselineCNN(backbone)

    # Load data
    train_loader, val_loader = data_loader(config)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(writer_path)  
    
    # Initialize model EMA, gradient scaler, optimizer, and scheduler
    ema = model_ema.ModelEmaV2(model, decay=config["PARAMETER"]["EMA_DECAY"]).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=config["PARAMETER"]["LEARNING_RATE"], weight_decay=config["PARAMETER"]["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["PARAMETER"]["MAX_LR"], epochs=train_epoch, steps_per_epoch=len(train_loader), three_phase=True)
    
    # Initialize best metric
    best_metric = 0
    
    # Move model to device
    model.to(device)
    evaluation(1, model, val_loader, writer, config=config)

    # Training loop
    for epoch in range(train_epoch):
        print(f'\nEpoch {epoch + 1}/{train_epoch}')
        print('-'*30)
        model.train()
        epoch_train(epoch, model, train_loader, scaler, optimizer, scheduler, ema, writer, config=config)
        
        model.eval()
        metric = evaluation(epoch, model, val_loader, writer, config=config)
        
        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric
        }
        torch.save(checkpoint, model_path)
        
        if (metric > best_metric):
            best_metric = metric
            checkpoint.update({"best_metric": best_metric})
            best_checkpoint_path = model_path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_checkpoint_path)
    
    writer.close()
    
    
if __name__ == "__main__":
    main()