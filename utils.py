
from datasets import load_dataset
import torch
from transformers import (
    AutoImageProcessor,
    ConvNextForImageClassification,
    TFConvNextForImageClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer, AutoFeatureExtractor, AutoModelForImageClassification, TFSwinForImageClassification
)

MODEL_PARAMS = {
    "cnn": {
        "name": "facebook/convnext-base-224-22k",
        "learningRate": 5e-5
    },
    "transformer": {
        "name": "microsoft/swin-base-patch4-window7-224-in22k",
        "learningRate": 1e-5
    }
}
TRAIN_BATCH_SIZE = 128
TRAIN_EPOCHS = 30
WEIGHT_DECAY = 1e-8


def preprocessing_fn(self, sample):
    sample['pixel_values'] = [
        self.image_processor(
            image if image.mode == 'RGB' else image.convert('RGB'),
            return_tensors='pt')['pixel_values'][0]
        for image in sample['image']
    ]
    return sample


def get_processed_dataset(full=False):
    """
    :return:
    """
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    data_size = dataset['train'].size if full else 1000
    processed_dataset = dataset['train']\
        .select(range(data_size))\
        .map(preprocessing_fn, remove_columns=['image'], num_proc=None, batched=True, batch_size=1000)

    return processed_dataset.train_test_split(test_size=0.15, stratify_by_column='label')


def get_model(model_name, framework):
    # Download pytorch model and preprocessor
    model = None
    if framework == 'tf':
        model = AutoModelForImageClassification.from_pretrained(model_name)
    elif framework == 'pt':
        # Download tensorflow model and preprocessor
        model = TFSwinForImageClassification.from_pretrained(model_name)
    else:
        print("ERROROR! Bad framework: ")
    return model


def get_acc_steps(model_type):
    num_gpus = torch.cuda.device_count()
    eff_batch_size = None
    if model_type == "cnn":
        eff_batch_size = 512
    elif model_type == "transformer":
        eff_batch_size = 1024
    else:
        print("BAD MODEL TYPE")
    return round(eff_batch_size / (num_gpus * TRAIN_BATCH_SIZE))

