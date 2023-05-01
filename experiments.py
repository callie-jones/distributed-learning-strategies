# ADD INFO

import torch
import tensorflow as tf
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    ConvNextForImageClassification,
    TFConvNextForImageClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer, AutoFeatureExtractor, AutoModelForImageClassification, TFSwinForImageClassification
)
import numpy as np
from PIL import Image
import tqdm
import timm.utils

# preprocess
def preprocessing_fn(sample, image_processor):
    sample['pixel_values'] = [
        image_processor(
            image if image.mode == 'RGB' else image.convert('RGB'),
            return_tensors='pt')['pixel_values'][0]
            for image in sample['image']
    ]
    return sample

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """
    Computes top-1 and top-5 accuracy in the training loop.
    """
    outputs = torch.from_numpy(eval_preds.predictions)
    targets = torch.from_numpy(eval_preds.label_ids)
    top1, top5 = timm.utils.accuracy(outputs, targets, topk=(1,5))
    return {'top_1_acc': top1, 'top_5_acc': top5}

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def get_processed_dataset(full=False):
    """
    :return:
    """
    # dataset = load_dataset("imagenet-1k", streaming=True, use_auth_token=True)
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    data_size = dataset['train'].size if full else 1000
    processed_dataset = dataset['train']\
        .select(range(data_size))\
        .map(preprocessing_fn, remove_columns=['image'],num_proc=None, batched=True, batch_size=1000)

    return processed_dataset.train_test_split(test_size=0.15, stratify_by_column='label')

def get_model(model_name, framework):
    # Download pytorch model and preprocessor
    model = None
    image_processor = AutoFeatureExtractor.from_pretrained()
    if framework == 'tf':
        model = AutoModelForImageClassification.from_pretrained(model_name)
    elif framework == 'pt':
        # Download tensorflow model and preprocessor
        model = TFSwinForImageClassification.from_pretrained(model_name)
    else:
        print("ERROROR! Bad framework: ")
    return model

def get_tf_training_args(model_type):
    eff_batch_size = None
    learning_rate = None
    train_batch_size = 128
    train_epochs = None
    weight_decay = None
    num_gpus = torch.cuda.device_count()
    if model_type == "cnn":
        eff_batch_size = 512
        learning_rate = 5e-5
        train_epochs = 30
        weight_decay = 1e-8
    elif model_type == "transformer":
        eff_batch_size = 1024
        learning_rate = 1e-5
        train_epochs = 30
        weight_decay = 1e-8
    else:
        print("BAD MODEL TYPE")
    acc_steps = round(eff_batch_size / (num_gpus * train_batch_size))
    return TrainingArguments(
        bf16=True,
        eval_accumulation_steps=4,
        evaluation_strategy='epoch',
        # eval_steps = 1,
        gradient_accumulation_steps=acc_steps,
        learning_rate=learning_rate,
        # cosine learning rate schedule?
        # layer-wise learning rate decay?
        logging_strategy='epoch',
        # logging_steps = 1,
        num_train_epochs=train_epochs,
        optim='adamw_torch',
        output_dir='training_output',
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        torch_compile=True,
        save_strategy='epoch',
        # save_steps = 10,
        save_total_limit=3,
        warmup_ratio=0.0,
        weight_decay=weight_decay
    )

def get_pt_training_args(model_type):
    eff_batch_size = None
    learning_rate = None
    train_batch_size = 128
    train_epochs = None
    weight_decay = None
    num_gpus = torch.cuda.device_count()
    if model_type == "cnn":
        eff_batch_size = 512
        learning_rate = 5e-5
        train_epochs = 30
        weight_decay = 1e-8
    elif model_type == "transformer":
        eff_batch_size = 1024
        learning_rate = 1e-5
        train_epochs = 30
        weight_decay = 1e-8
    else:
        print("BAD MODEL TYPE")
    acc_steps = round(eff_batch_size / (num_gpus * train_batch_size))
    return TrainingArguments(
        bf16=True,
        eval_accumulation_steps=4,
        evaluation_strategy='epoch',
        # eval_steps = 1,
        gradient_accumulation_steps=acc_steps,
        learning_rate=learning_rate,
        # cosine learning rate schedule?
        # layer-wise learning rate decay?
        logging_strategy='epoch',
        # logging_steps = 1,
        num_train_epochs=train_epochs,
        optim='adamw_torch',
        output_dir='training_output',
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        torch_compile=True,
        save_strategy='epoch',
        # save_steps = 10,
        save_total_limit=3,
        warmup_ratio=0.0,
        weight_decay=weight_decay
    )


def run_experiment(model_name, model_type, framework, debug_mode=True):
    """
    info
    :return:
    """
    # get dataset
    processed_dataset = get_processed_dataset()
    # torch.cuda.device_count()
    # get model
    model = get_model(model_name, framework)
    # get training arguments
    training_args = get_pt_training_args(model_type) if framework == "pt" else get_tf_training_args(model_type)
    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        compute_metrics=compute_metrics
    )
    trainer.train()
    accuracy(torch.from_numpy(output), torch.from_numpy(target), topk=(1, 5))




if __name__ == "__main__":
    model_name = "microsoft/swin-base-patch4-window7-224-in22k"
    model_type = "cnn"
    framework = "pt"
    # run single model
    # run all models
    run_experiment(model_name, model_type, framework)

