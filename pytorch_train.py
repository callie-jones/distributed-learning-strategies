import argparse
import os
import json
from typing import Tuple, Union
import time
from tqdm import tqdm
import torch
import wandb
import timm.utils
import utils
from PIL import Image
import random
from transformers import AutoImageProcessor

class TorchImagenet(torch.utils.data.Dataset):

    def __init__(self, root: os.PathLike, split: str, processor: AutoImageProcessor):
        self.samples = []
        self.targets = []
        self.processor = processor

        with open(os.path.join(root, 'imagenet_class_index.json'), 'rb') as f:
            json_file = json.load(f)
            images_by_class = {int(class_id): [] for class_id in json_file}
            self.syn_to_class = {v[0]: int(class_id) for class_id, v in json_file.items()}
        
        with open(os.path.join(root, 'ILSVRC2012_val_labels.json'), 'rb') as f:
            self.img_to_syn = json.load(f)

        # Divide images by class to take stratified sample
        samples_dir = os.path.join(root, 'img') 
        for entry in os.listdir(samples_dir):
            target = self.syn_to_class[ self.img_to_syn[entry] ]
            images_by_class[target].append(entry)

        # Sample images in equal proportion by class
        samples_per_class = 10 if split == 'train' else 3
        random.seed(42)

        for class_id, class_entries in images_by_class.items():
            img_samples = random.sample(class_entries, samples_per_class)
            img_samples = [os.path.join(samples_dir, img) for img in img_samples]
            self.samples += img_samples
            self.targets += [class_id]*len(img_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        pixel_values = self.processor(img, return_tensors='pt')['pixel_values']
        return pixel_values, self.targets[idx]

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, criterion: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader, rank
) -> Tuple[float, float]:

    running_vloss, running_vacc = 0.0, 0.0
    for pixel_values, labels in tqdm(test_dataloader):

        # Make predictions on validation dataset
        labels = labels.to(rank, non_blocking=True)
        pixel_values = torch.squeeze(pixel_values)
        pixel_values = pixel_values.to(rank, non_blocking=True)

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

@torch.compile(mode='default')
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
    optimizer: torch.optim.Optimizer, rank, 
    grad_scaler: torch.cuda.amp.GradScaler
) -> Tuple[float, float]:
    
    running_loss, running_acc = 0.0, 0.0
    for pixel_values, labels in tqdm(train_dataloader):

        labels = labels.to(rank, non_blocking=True)
        pixel_values = torch.squeeze(pixel_values)
        pixel_values = pixel_values.to(rank, non_blocking=True)

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
    rank, model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel], 
    epochs: int, model_name: str, train_dataloader: torch.utils.data.DataLoader,
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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define objects needed for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=utils.MODEL_PARAMS[model_name]['learningRate'], weight_decay=utils.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(rank)
    grad_scaler = torch.cuda.amp.GradScaler()

    # Compile model and put on GPU
    model = torch.compile(model, mode='default', fullgraph=False)

    metrics = {}

    for epoch in range(epochs):

        # Need to tell DistributedSampler what epoch it is
        # if torch.cuda.device_count() > 1:
        train_dataloader.sampler.set_epoch(epoch)
        # test_dataloader.sampler.set_epoch(epoch)
        
        # Train for one epoch and get loss
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, criterion, train_dataloader, 
                                                optimizer, rank, grad_scaler)
        epoch_end_time = time.time()
        epoch_train_time = epoch_end_time - epoch_start_time
        
        metrics['time_per_epoch'] = epoch_train_time
        metrics['train_loss'] = train_loss
        metrics['train_accuracy'] = train_acc

        # Evaluate on test set
        eval_start_time = time.time()
        val_loss, val_accuracy = evaluate(model, criterion, test_dataloader, rank)
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

    if torch.cuda.device_count() > 1:
        cleanup()

    return metrics

def prepare(rank, world_size):
    image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k')
    image_processor.do_resize = True

    train_dataset = TorchImagenet(
        root = 'imagenet_val/',
        split = 'train',
        processor = image_processor
    )

    test_dataset = TorchImagenet(
        root = 'imagenet_val/',
        split = 'val',
        processor = image_processor
    )

    train_sampler, test_sampler = None, None 
    # num_workers = 1 if torch.cuda.device_count() > 1 else 8
    # pin_memory = False if torch.cuda.device_count() > 1 else True

    # Need DistributedSampler if training with more than 1 GPU       
    # if torch.cuda.device_count() > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas = world_size, rank = rank,
        shuffle = True, drop_last = False
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas = world_size, rank = rank,
        shuffle = False, drop_last = False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 128, num_workers = 8, 
        pin_memory = False, prefetch_factor = 2,
        persistent_workers = True, sampler = train_sampler
    )

    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 128, num_workers = 8,
    pin_memory = False, prefetch_factor = 2, sampler = test_sampler
    )

    return train_dataloader, test_dataloader


def setup(rank, world_size):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    train_dataloader, test_dataloader = prepare(rank, world_size)
    model = utils.get_model('cnn', 'pt').to(rank)

    # if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids = [rank], output_device = rank,
        find_unused_parameters = False
    )

    train(rank, model, 5, 'cnn', train_dataloader, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for training ConvNext and Swin models in PyTorch.")
    parser.add_argument('model', choices = ['cnn', 'transformer'],
                        help = 'the model to train eg. "cnn" or "transformer"')
    parser.add_argument('training_strategy', choices = ['ddp', 'fsdp'],
                        help = 'the distriuted training strategy eg. "ddp" or "fsdp"')
    parser.add_argument('debug', choices = ['debug', 'prod'],
                        help = '"debug" to train for one epoch on small data else "prod" to do full training')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args = (world_size, ),
        nprocs = world_size
    )