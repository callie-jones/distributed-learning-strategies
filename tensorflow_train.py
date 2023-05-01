import sys

import tensorflow as tf

import utils
from tensorflow_gradient_accumulator import GradientAccumulator


class DistributedLearningTensorFlow:
    def __init__(self):
        pass

    def run_experiment(self, model_type, debug_mode=True):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            EPOCHS = 1 if debug_mode else utils.TRAIN_EPOCHS
            model_name = utils.MODEL_PARAMS[model_type].name

            # get the dataset
            processed_dataset = utils.get_processed_dataset()
            tf_train_dataset = processed_dataset['train'].to_tf_dataset(batch_size=utils.TRAIN_BATCH_SIZE)
            tf_test_dataset = processed_dataset['test'].to_tf_dataset(batch_size=utils.TRAIN_BATCH_SIZE)
            model = utils.get_model(model_name, "tf")

            # compile the model
            optimizer = tf.keras.optimizers.Adam(
                              learning_rate=utils.MODEL_PARAMS[model_type].learningRate,
                              weight_decay=utils.WEIGHT_DECAY
                        )
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=GradientAccumulator(optimizer, accum_steps=utils.get_acc_steps(model_type)),
                          metrics=['accuracy'])

            # train the model
            model.fit(tf_train_dataset, epochs=EPOCHS)

            # evaluate the model
            model.load_weights(tf.train.latest_checkpoint("./training_checkpoints"))
            eval_loss, eval_acc = model.evaluate(tf_test_dataset)
            print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))


if __name__ == '__main__':
    mt = sys.argv[1] if len(sys.argv) == 3 else "cnn"
    DistributedLearningTensorFlow().run_experiment(mt)

