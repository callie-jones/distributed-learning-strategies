import argparse
import numpy as np
from PIL import Image
import tqdm
import timm.utils

import datasets
from transformers import (
    AutoImageProcessor,
    ConvNextForImageClassification,
    TFConvNextForImageClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer, AutoFeatureExtractor, AutoModelForImageClassification, TFSwinForImageClassification
)

import utils

def main(args: argparse.Namespace):
    debug = True if args.debug == 'debug' else False

    dataset = utils.get_processed_data(args.model, debug, 'pt')
    print(dataset)

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