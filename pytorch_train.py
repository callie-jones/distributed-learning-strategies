import argparse
# import numpy as np
# from PIL import Image
import tqdm
import timm.utils
import torch
import datasets
from transformers import (
    ConvNextForImageClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer,
)

import utils

def set_layerwise_lr_decay(model: ConvNextForImageClassification) -> ConvNextForImageClassification:
    """
    Sets layer-wise learning rate decay for ConvNext models in accordance with how the 
    original authors did so at: https://github.com/facebookresearch/ConvNeXt/tree/main
    """
    raise NotImplementedError

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """
    Computes top-1 and top-5 accuracy in the training loop.
    """
    outputs = torch.from_numpy(eval_preds.predictions)
    targets = torch.from_numpy(eval_preds.label_ids)
    top1, top5 = timm.utils.accuracy(outputs, targets, topk=(1,5))
    return {'top_1_acc': top1, 'top_5_acc': top5}

def main(args: argparse.Namespace):
    debug = True if args.debug == 'debug' else False

    dataset = utils.get_processed_data(args.model, debug, 'pt')
    model = utils.get_model(args.model, 'pt')

    # if args.model == 'cnn':
    #     model = set_layerwise_lr_decay(model)

    training_args = TrainingArguments(
    
        optim = 'adamw_torch',
        num_train_epochs = 1 if debug else 30,
        learning_rate = utils.MODEL_PARAMS[args.model]['learningRate'],
        # cosine learning rate schedule?
        # layer-wise learning rate decay?
        warmup_ratio = 0.0,
        per_device_train_batch_size = 128,
        gradient_accumulation_steps = utils.get_acc_steps(args.model),
        per_device_eval_batch_size = 128,
        eval_accumulation_steps = utils.get_acc_steps(args.model),
        weight_decay = 1e-8,
        fp16 = True,
        torch_compile = True if torch.cuda_is_available() else False,

        evaluation_strategy = 'epoch',
        logging_strategy = 'epoch',
        output_dir = f'training_output/{args.model}/pt/{torch.cuda.device_count()}_gpus',
        save_strategy = 'epoch',
        save_total_limit = 2
    )

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    compute_metrics = compute_metrics
    )

    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for training ConvNext and Swin models in PyTorch.")
    parser.add_argument('model', choices = ['cnn', 'transformer'],
                        help = 'the model to train eg. "cnn" or "transformer"')
    parser.add_argument('training_strategy', choices = ['ddp', 'fsdp'],
                        help = 'the distriuted training strategy eg. "ddp" or "fsdp"')
    parser.add_argument('debug', choices = ['debug', 'prod'],
                        help = '"debug" to train for one epoch on small data else "prod" to do full training')
    args = parser.parse_args()

    main(args)