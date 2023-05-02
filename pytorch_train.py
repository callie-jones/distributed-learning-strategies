import argparse
import os
import numpy as np
import torch
import wandb
import datasets
from transformers import (
    EvalPrediction,
    TrainingArguments,
    Trainer,
)
import utils

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """
    Computes top-1 and top-5 accuracy in the training loop.
    """
    predicted_labels = np.argmax(eval_preds.predictions, axis=1)
    correct_preds = np.sum(predicted_labels == eval_preds.label_ids)
    accuracy = correct_preds / len(eval_preds.label_ids)
    return {'accuracy': accuracy}

def main(args: argparse.Namespace):
    debug = True if args.debug == 'debug' else False

    dataset = utils.get_processed_data(args.model, debug, 'pt')
    model = utils.get_model(args.model, 'pt')
    model_name_short = utils.MODEL_PARAMS[args.model]['name_short']

    os.environ['WANDB_PROJECT'] = 'ml-framework-benchmarking'
    os.environ['WANDB_LOG_MODEL'] = 'false'
    os.environ['WANDB_WATCH'] = 'false'

    training_args = TrainingArguments(
    
        optim = 'adamw_torch',
        num_train_epochs = 2 if debug else 30,
        learning_rate = utils.MODEL_PARAMS[args.model]['learningRate'],
        warmup_ratio = 0.0,
        per_device_train_batch_size = int(128 / torch.cuda.device_count()),
        # gradient_accumulation_steps = utils.get_acc_steps(args.model),
        per_device_eval_batch_size = int(128 / torch.cuda.device_count()),
        # eval_accumulation_steps = utils.get_acc_steps(args.model),
        weight_decay = 1e-8,
        ddp_find_unused_parameters = False,
        ddp_bucket_cap_mb = 25,
        fp16 = True,
        torch_compile = True if torch.cuda.is_available() else False,

        evaluation_strategy = 'epoch',
        logging_strategy = 'epoch',
        output_dir = f'training_output/{args.model}/pt/{torch.cuda.device_count()}_gpus',
        save_strategy = 'epoch',
        save_total_limit = 2,
        report_to = 'wandb',
        run_name = f'{model_name_short}_pytorch_{args.training_strategy}_{torch.cuda.device_count()}_gpu(s)'
    )

    print(f"\nBeginning {args.model} training in {args.debug} mode.\n")

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    compute_metrics = compute_metrics
    )

    trainer.train()
    wandb.finish()
    

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