import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional.segmentation import generalized_dice_score as dice

def hardunet_train_loop(model: nn.Module, 
                  optim: optim, 
                  loss_fn: F, 
                  device: torch.device,
                  train_data: DataLoader, 
                  eval_data: DataLoader = None, 
                  epochs=1001,
                  scheduler: torch.optim.lr_scheduler = None,
                  checks=100):
    
    model = model.to(device)
    n_classes = model.get_classes()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_data:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            train_dice = sum(dice(y_pred, batch_y, n_classes)) / len(batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        if scheduler is not None:
            scheduler.step()

        # Validation phase
        if eval_data is not None:
            model.eval()
            val_losses = []
            val_dices = []
            with torch.inference_mode():
                for val_X, val_y in eval_data:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    val_pred = model(val_X)
                    val_loss = loss_fn(val_pred, val_y)
                    val_dice = sum(dice(val_pred, val_y, n_classes)) / len(val_y)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
        
        if epoch % checks == 0:
            print(f"Epoch: {epoch}")
            print(f"TRAIN => Loss: {loss.item():.4f} | Average Dice: {train_dice:.4f}")
            if eval_data is not None:
                print(f"EVAL => Loss: {val_loss.item():.4f} | Average Dice: {val_dice:.4f}")
                print(f"Average Val Loss: {avg_val_loss}")
            print("----------------------------------------------------")
    