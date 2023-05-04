import sys
import os
import argparse
import tensorflow as tf
import torch.cuda

import utils
import wandb
from wandb.keras import WandbMetricsLogger
import tensorflow_datasets as tfds
from tfswin import preprocess_input


class DistributedLearningTensorFlow:
    def __init__(self):
        pass

    def get_transformer_data(self):
        isSaved = True if os.path.exists(f'imagenet2012.tfrecord') else False
        print(f"\nDownloading dataset: {isSaved}")
        processed_dataset = tfds.load('imagenet2012', split='validation', shuffle_files=True, download=isSaved)
        processed_dataset = processed_dataset.map(self.preprocessing_fn, num_parallel_calls=tf.data.AUTOTUNE)
        processed_dataset = processed_dataset.batch(utils.TRAIN_BATCH_SIZE)
        return processed_dataset

    def preprocessing_fn(self, sample, input_size=224, crop_pct=0.875):
        print("\npre-processing TensorFlow data")
        scale_size = tf.math.floor(input_size / crop_pct)

        image = sample['image']

        shape = tf.shape(image)[:2]
        shape = tf.cast(shape, 'float32')
        shape *= scale_size / tf.reduce_min(shape)
        shape = tf.round(shape)
        shape = tf.cast(shape, 'int32')

        image = tf.image.resize(image, shape, method=tf.image.ResizeMethod.BICUBIC)
        image = tf.round(image)
        image = tf.clip_by_value(image, 0., 255.)
        image = tf.cast(image, 'uint8')

        pad_h, pad_w = tf.unstack((shape - input_size) // 2)
        image = image[pad_h:pad_h + input_size, pad_w:pad_w + input_size]

        image = preprocess_input(image)

        return image, sample['label']

    def run_experiment(self, args):
        print(f"\nBeginning {args.model} training in {args.debug} mode.\n")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            EPOCHS = 1 if args.debug else utils.TRAIN_EPOCHS

            processed_dataset = self.get_transformer_data() if args.model == "transformer" else utils.get_processed_data(args.model, args.debug, "tf")

            print("\nRetrieve model")
            model = utils.get_model(args.model, "tf")

            # compile the model
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=utils.MODEL_PARAMS[args.model]["learningRate"],
                weight_decay=utils.WEIGHT_DECAY
            )
            # GradientAccumulator(optimizer, accum_steps=get_acc_steps(model_type))
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("\nCompile Model")
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=optimizer,
                          metrics=['accuracy'])

            # train the model
            run = wandb.init(
                project='ml-framework-benchmarking',
                name=f'{utils.MODEL_PARAMS[args.model]["name_short"]}_tensorflow_{torch.cuda.device_count()}_gpu(s)'
            )
            model.fit(processed_dataset["train"], validation_data=processed_dataset["validation"],
                      epochs=EPOCHS)  # steps_per_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for training ConvNext and Swin models in TensorFlow.")
    parser.add_argument('model', choices=['cnn', 'transformer'],
                        help='the model to train eg. "cnn" or "transformer"')
    parser.add_argument('debug', choices=['debug', 'prod'],
                        help='"debug" to train for one epoch on small data else "prod" to do full training')
    parsed_args = parser.parse_args()
    DistributedLearningTensorFlow().run_experiment(parsed_args)

