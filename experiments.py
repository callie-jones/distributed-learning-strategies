import sys

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

class DistributedLearningStrategies:
    def __init__(self):
        self.image_processor = None

    def preprocessing_fn(self, sample):
        sample['pixel_values'] = [
            self.image_processor(
                image if image.mode == 'RGB' else image.convert('RGB'),
                return_tensors='pt')['pixel_values'][0]
                for image in sample['image']
        ]
        return sample

    def compute_metrics(self, eval_preds: EvalPrediction) -> dict:
        """
        Computes top-1 and top-5 accuracy in the training loop.
        """
        outputs = torch.from_numpy(eval_preds.predictions)
        targets = torch.from_numpy(eval_preds.label_ids)
        top1, top5 = timm.utils.accuracy(outputs, targets, topk=(1,5))
        return {'top_1_acc': top1, 'top_5_acc': top5}

    def get_processed_dataset(self, full=False):
        """
        :return:
        """
        # dataset = load_dataset("imagenet-1k", streaming=True, use_auth_token=True)
        dataset = load_dataset("mrm8488/ImageNet1K-val")
        data_size = dataset['train'].size if full else 1000
        processed_dataset = dataset['train']\
            .select(range(data_size))\
            .map(self.preprocessing_fn, remove_columns=['image'], num_proc=None, batched=True, batch_size=1000)

        return processed_dataset.train_test_split(test_size=0.15, stratify_by_column='label')

    def get_model(self, model_name, framework):
        # Download pytorch model and preprocessor
        model = None
        self.image_processor = AutoFeatureExtractor.from_pretrained(model_name)
        if framework == 'tf':
            model = AutoModelForImageClassification.from_pretrained(model_name)
        elif framework == 'pt':
            # Download tensorflow model and preprocessor
            model = TFSwinForImageClassification.from_pretrained(model_name)
        else:
            print("ERROROR! Bad framework: ")
        return model

    def get_pt_training_args(self, model_type):
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

    def run_pt_experiment(self, model_type, debug_mode=True):
        """
        info
        :return:
        """
        # get dataset
        model_name = "facebook/convnext-base-224-22k" if model_type == "cnn" else "microsoft/swin-base-patch4-window7-224-in22k"
        processed_dataset = self.get_processed_dataset()
        # torch.cuda.device_count()
        # get model
        model = self.get_model(model_name, "pt")
        # get training arguments
        training_args = self.get_pt_training_args(model_type)
        # train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset['test'],
            compute_metrics=self.compute_metrics
        )
        trainer.train()

    def run_tf_experiment(self, model_type):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            EPOCHS = 30
            model_name = "facebook/convnext-base-224-22k" if model_type == "cnn" else "microsoft/swin-base-patch4-window7-224-in22k"
            # get the dataset
            processed_dataset = self.get_processed_dataset()
            tf_train_dataset = processed_dataset['train'].to_tf_dataset(batch_size=128)
            tf_test_dataset = processed_dataset['test'].to_tf_dataset(batch_size=128)
            model = self.get_model(model_name, "tf")

            # compile the model
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['accuracy'])
            # train the model
            model.fit(tf_train_dataset, epochs=EPOCHS)

            # evaluate the model
            model.load_weights(tf.train.latest_checkpoint("./training_checkpoints"))
            eval_loss, eval_acc = model.evaluate(tf_test_dataset)
            print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))

    def run_all_experiments(self):
        print("==== CNN PyTorch Experiment ====")
        self.run_pt_experiment("cnn")

        print("\n\n==== CNN TensorFlow Experiment ====")
        self.run_tf_experiment("cnn")

        print("\n\n==== Transformer PyTorch Experiment ====")
        self.run_pt_experiment("transformer")

        print("\n\n==== Transformer TensorFlow Experiment ====")
        self.run_tf_experiment("transformer")

if __name__ == "__main__":
    mt = sys.argv[1] if len(sys.argv) == 3 else "cnn"
    fw = sys.argv[2] if len(sys.argv) == 3 else "pt"
    # run single model
    # run all models
    # debug mode = true -> small epoch = 1
    DistributedLearningStrategies().run_experiment(mt, fw)

