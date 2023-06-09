import os
import argparse
import tensorflow as tf

import utils
import wandb
from wandb.keras import WandbMetricsLogger

class DistributedLearningTensorFlow:
    def get_transformer_data(self):
        isSaved = not os.path.exists(f'imagenet2012.tfrecord')
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
            EPOCHS = utils.TRAIN_EPOCHS

            if args.model == "transformer":
                processed_dataset_train = self.get_transformer_data()
            else:
                debugMode = args.debug == "debug"
                processed_dataset_train, processed_dataset_validation = utils.get_processed_data(args.model, debugMode, "tf")

            print("\nRetrieve model")
            # model = utils.get_model(args.model, "tf")
            model = utils.get_model(args.model, "tf")
            print(f"trainable weights = {model.summary()}")

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
                          jit_compile=True,
                          metrics=['accuracy'])

            # train the model
            run = wandb.init(
                project='ml-framework-benchmarking',
                name=f'{utils.MODEL_PARAMS[args.model]["name_short"]}_tensorflow_{len(tf.config.list_physical_devices("GPU"))}_gpu(s)')
            model.fit(processed_dataset_train, validation_data=processed_dataset_validation,
                      epochs=EPOCHS)  # steps_per_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for training ConvNext and Swin models in TensorFlow.")
    parser.add_argument('model', choices=['cnn', 'transformer'],
                        help='the model to train eg. "cnn" or "transformer"')
    parser.add_argument('debug', choices=['debug', 'prod'],
                        help='"debug" to train for one epoch on small data else "prod" to do full training')
    parsed_args = parser.parse_args()
    DistributedLearningTensorFlow().run_experiment(parsed_args)

