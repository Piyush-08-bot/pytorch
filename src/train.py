"""Training logic."""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
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


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
