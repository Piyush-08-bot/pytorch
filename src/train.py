"""Training logic."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_epoch(model: nn.Module, train_loader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch.
    
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model: nn.Module, val_loader, criterion: nn.Module, 
             device: torch.device) -> float:
    """Evaluate model on validation set.
    
    Returns:
        Average loss over the validation set.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
