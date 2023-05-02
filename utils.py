import json
import datasets
import torch
from transformers import (
    AutoImageProcessor,
    ConvNextForImageClassification,
    TFConvNextForImageClassification,
    SwinForImageClassification,
    TFSwinForImageClassification,
    PreTrainedModel,
    EvalPrediction,
    TrainingArguments,
)

FRAMEWORKS = ['pt', 'tf']
DATASETS = {
    "debug": "mrm8488/ImageNet1K-val",
    "prod": "imagenet-1k"
}
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


def get_processed_data(model_type: str, debug: bool, framework: str):
    """
    Load the processor for Swin/ConvNext and the data. Preprocess the data using the
    processor and return it. 

    :param: model_type (str) One of 'cnn' or 'transformer'. Used to load the appropriate processor.
    :param: debug (bool) One of 'debug' or 'prod'. If 'prod', use full Imagenet. If 'debug',
                         use only a subset of the validation split.
    :param: framework (str) One of 'pt' or 'tf'. If 'tf' convert to TensorFlow dataset.
    :returns: The training data with 'train' and 'test' splits. If framework is 'pt', returns a 
              datasets.DatasetDict. If framework is 'tf', returns a tuple (train_ds, test_ds)
              of TensorFlow datasets that can be passed directly to model.fit().
    """
    if model_type not in MODEL_PARAMS:
        raise ValueError("Argument model_type must be one of 'cnn', 'transformer'. You supplied:", model_type)
    if framework not in FRAMEWORKS:
        raise ValueError("Argument framework must be one of 'pt', 'tf'. You supplied:", framework)

    # Download the image processor
    processor_name = MODEL_PARAMS[model_type]['name']
    print(f"\nDownloading image processor {processor_name} from Hugging Face hub.\n")
    processor = AutoImageProcessor.from_pretrained(processor_name)

    # Download the dataset (takes a while if using full dataset)
    ds_name = DATASETS['debug'] if debug == True else DATASETS['prod']
    print(f"\nDownloading dataset {ds_name} from Hugging Face hub.\n")
    raw_dataset = datasets.load_dataset(ds_name)

    # Define function to preprocess images
    def preprocessing_fn(sample, processor=processor):
        sample['pixel_values'] = [
            processor(image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
                for image in sample['image']
        ]
        return sample
    
    # Preprocess images (takes a while if using full dataset)
    print("\nPreprocessing training and validation data.\n")
    if debug == True:
        processed_dataset = raw_dataset['train'].select(range(1000)) \
                                       .map(preprocessing_fn, remove_columns=['image'],
                                            num_proc=None, batched=True, batch_size=1000
                                        ).train_test_split(test_size=0.15,
                                                           stratify_by_column='label')
    else:
        processed_dataset = raw_dataset.map(preprocessing_fn, remove_columns=['image'],
                                            num_proc=2, batched=True, batch_size=1500
                                        ).train_test_split(
                                            test_size = 0.15,
                                            stratify_by_column = 'label')
        
    if framework == 'tf':
        tf_train_dataset = processed_dataset['train'].to_tf_dataset(columns='pixel_values', label_cols='label',
                                                                    batch_size=TRAIN_BATCH_SIZE)
        tf_test_dataset = processed_dataset['test'].to_tf_dataset(columns='pixel_values', label_cols='label',
                                                                  batch_size=TRAIN_BATCH_SIZE, shuffle=False)
        processed_dataset = (tf_train_dataset, tf_test_dataset)

    print('\n', processed_dataset, '\n')
    return processed_dataset

def get_model(model_type: str, framework: str) -> PreTrainedModel:
    """
    Return the specified model for the specified framework.

    :param: model_name (str) 'cnn' for ConvNext or 'transformer' for Swin.
    :param: framework (str) One of 'pt' or 'tf'. Specifies model framework.
    :returns: (transformers.PreTrainedModel) The specified model from HF.
    """
    if model_type not in MODEL_PARAMS:
        raise ValueError("Argument model_type must be one of 'cnn', 'transformer'. You supplied:", model_type)
    if framework not in FRAMEWORKS:
        raise ValueError("Argument framework must be one of 'pt', 'tf'. You supplied:", framework)

    model = None
    model_name = MODEL_PARAMS[model_type]['name']

    # Download the model from Hugging Face hub
    print(f"\nDownloading {model_type} for {framework} framework.\n")
    if model_type == 'cnn' and framework == 'tf':
        model = TFConvNextForImageClassification.from_pretrained(model_name)
    elif model_type == 'cnn' and framework == 'pt':
        model = ConvNextForImageClassification.from_pretrained(model_name)
    elif model_type == 'transformer' and framework == 'tf':
        model = TFSwinForImageClassification.from_pretrained(model_name)
    elif model_type == 'transformer' and framework == 'pt':
        model = SwinForImageClassification.from_pretrained(model_name)

    # Adjust some configuration settings
    with open('id2label.json', 'r') as f:
        id2label = {int(id): label for id, label in json.load(f).items()}

    model.config.id2label = id2label
    model.config.label2id = {label: id for id, label in id2label.items()}
    model.config.num_labels = 1000

    return model

def get_acc_steps(model_type: str):

    if model_type not in MODEL_PARAMS:
        raise ValueError("Argument model_type must be one of 'cnn', 'transformer'. You supplied:", model_type)
        
    num_gpus = torch.cuda.device_count()
    eff_batch_size = None
    if model_type == "cnn":
        eff_batch_size = 512
    elif model_type == "transformer":
        eff_batch_size = 1024
    else:
        print("BAD MODEL TYPE")
    return round(eff_batch_size / (num_gpus * TRAIN_BATCH_SIZE))

