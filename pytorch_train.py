import argparse
import os
from typing import Tuple
import time
from tqdm import tqdm
import numpy as np
import torch
import wandb
import datasets
import timm.utils
import utils


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, criterion: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float]:

    running_vloss, running_vacc = 0.0, 0.0
    for val_batch in tqdm(test_dataloader):

        # Make predictions on validation dataset
        labels = val_batch['label'].to(device, non_blocking=True)
        pixel_values = val_batch['pixel_values'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            val_outputs = model(pixel_values).logits
            val_loss = criterion(val_outputs, labels)
        
        running_vloss += val_loss

        # Compute accuracy for this batch
        acc1, _ = timm.utils.accuracy(val_outputs, labels, topk=(1, 5))
        running_vacc += acc1 / 100

    avg_vloss = running_vloss / len(test_dataloader)
    avg_vacc = running_vacc / len(test_dataloader)
    
    return avg_vloss, avg_vacc

@torch.compile(mode='max-autotune')
def train_one_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    pixel_values: torch.Tensor, labels: torch.Tensor, 
    criterion: torch.nn.Module, grad_scaler: torch.cuda.amp.GradScaler
) -> Tuple[float, torch.Tensor]:
    # Compute output and loss with mixed precision
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(pixel_values).logits
        loss = criterion(output, labels)

    # loss_value = loss.item()
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss.item(), output

def train_one_epoch(
    model: torch.nn.Module, criterion: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, device: torch.device, 
    grad_scaler: torch.cuda.amp.GradScaler
) -> Tuple[float, float]:
    
    running_loss, running_acc = 0.0, 0.0
    for batch in tqdm(train_dataloader):
        labels = batch['label'].to(device, non_blocking=True)
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)

        # Compute output and loss with mixed precision
        batch_loss, output = train_one_step(
                model, optimizer, pixel_values, 
                labels, criterion, grad_scaler)

        # Add to running accuracy and loss
        acc1, _ = timm.utils.accuracy(output, labels, topk=(1, 5))
        running_acc += acc1 / 100
        running_loss += batch_loss

    avg_loss = running_loss / len(train_dataloader)
    avg_acc = running_acc / len(train_dataloader)

    return avg_loss, avg_acc

def train(
    model: torch.nn.Module, epochs: int, model_name: str,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader
    ):
    
    # Configure experiment tracking
    os.environ['WANDB_PROJECT'] = 'ml-framework-benchmarking'
    os.environ['WANDB_LOG_MODEL'] = 'false'
    os.environ['WANDB_WATCH'] = 'false'

    log_filename = f"{utils.MODEL_PARAMS[model_name]['name_short']}_pytorch_ddp_{torch.cuda.device_count()}_gpu(s)"
    run = wandb.init(
        project = 'ml-framework-benchmarking',
        name = log_filename
    )

    # Remove old log file if it exists
    if os.path.exists(log_filename):
        os.remove(log_filename)

    torch.backends.cudnn.benchmark = True

    # Define objects needed for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=utils.MODEL_PARAMS[model_name]['learningRate'], weight_decay=utils.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    grad_scaler = torch.cuda.amp.GradScaler()

    # Compile model and put on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = torch.compile(model, mode='max-autotune', fullgraph=True)

    metrics = {}

    for epoch in range(epochs):
        
        # Train for one epoch and get loss
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, criterion, train_dataloader, 
                                                optimizer, epoch, device, grad_scaler)
        epoch_end_time = time.time()
        epoch_train_time = epoch_end_time - epoch_start_time
        
        metrics['time_per_epoch'] = epoch_train_time
        metrics['train_loss'] = train_loss
        metrics['train_accuracy'] = train_acc

        # Evaluate on test set
        eval_start_time = time.time()
        val_loss, val_accuracy = evaluate(model, criterion, test_dataloader, device)
        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time

        metrics['val_time'] = eval_time
        metrics['val_accuracy'] = val_accuracy
        metrics['val_loss'] = val_loss
        metrics['epoch'] = epoch + 1

        # Log metrics in file, terminal output, and wandb
        epoch_metrics = f"epoch {epoch+1}: LOSS train {train_loss:.4f} val {val_loss:.4f} ACCURACY train {train_acc:.4f} val {val_accuracy:.4f} TIME train {epoch_train_time:.4f} eval {eval_time:.4f}"
        print(epoch_metrics)
        with open(log_filename, 'a') as f:
            f.write(epoch_metrics + '\n')
        run.log(metrics)
        
    run.finish()
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for training ConvNext and Swin models in PyTorch.")
    parser.add_argument('model', choices = ['cnn', 'transformer'],
                        help = 'the model to train eg. "cnn" or "transformer"')
    parser.add_argument('training_strategy', choices = ['ddp', 'fsdp'],
                        help = 'the distriuted training strategy eg. "ddp" or "fsdp"')
    parser.add_argument('debug', choices = ['debug', 'prod'],
                        help = '"debug" to train for one epoch on small data else "prod" to do full training')
    args = parser.parse_args()

    debug = True if args.debug == 'debug' else False

    train_dataloader, val_dataloader = utils.get_processed_data(args.model, debug, 'pt')
    model = utils.get_model(args.model, 'pt')

    train(model, 2, args.model, train_dataloader, val_dataloader)