# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked
import numpy as np


@tf.keras.utils.register_keras_serializable(package="Addons")
class GradientAccumulator(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        accum_steps: types.TensorLike = 4,
        name: str = "GradientAccumulator",
        **kwargs,
    ):
        r"""Construct a new GradientAccumulator optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulator".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._optimizer = tf.keras.optimizers.get(optimizer)
        self._gradients = []
        self._accum_steps = accum_steps

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad, use_locking=self._use_locking, read_value=False
            )

        def _apply():
            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var, apply_state=apply_state
                )
            else:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def _resource_apply_sparse(self, grad: types.TensorLike, var, indices, apply_state):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self._optimizer._sparse_apply_args:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices), var, indices
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        config = {
            "accum_steps": self._accum_steps,
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

def get_ffn_model(input_size: int, output_size: int, hidden_size: int = 64) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_size,))
    x = inputs
    x = tf.keras.layers.Dense(units=hidden_size, activation='tanh')(x)
    x = tf.keras.layers.Dense(units=hidden_size, activation='tanh')(x)
    x = tf.keras.layers.Dense(units=output_size, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def make_dataset(inputs, targets, batch_size: int, split: str, limit: int = None):
    def sample_generator_():
        while True:
            idx = np.random.randint(0, len(inputs))
            yield inputs[idx].flatten(), tf.one_hot(targets[idx], depth=num_classes)

    assert split in ('train', 'test', 'dev'), \
        f'Split must be one of "train", "test" or "dev". Got: {split}'

    inputs = inputs.astype(np.float32) / 255.0
    inputs = np.expand_dims(inputs, axis=-1)
    num_classes = len(set(targets))

    input_shape = (np.prod(inputs[0].shape),)
    target_shape = (num_classes,)

    dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator_(),
        output_types=(tf.float32, tf.float32),
        output_shapes=(input_shape, target_shape)
    )

    is_training = split == 'train'

    if is_training:
        dataset = dataset.repeat()

    if limit:
        dataset = dataset.take(limit)

    return dataset.padded_batch(batch_size)


def main():
    train_batch_size = 1
    valid_batch_size = 10
    grad_acc_n = 4
    steps_per_epoch = 1000 * grad_acc_n  # Make sure we have the same number of updates

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = make_dataset(x_train, y_train, batch_size=train_batch_size, split='train')
    valid_data = make_dataset(x_test, y_test, batch_size=valid_batch_size, split='dev', limit=500)

    input_size = train_data.element_spec[0].shape[-1]
    output_size = train_data.element_spec[1].shape[-1]

    epochs = 2

    for precision_policy in ['float32', 'mixed_float16']:
        print('#' * 72)
        print(f'Setting precision-policy to "{precision_policy}"')

        tf.keras.mixed_precision.set_global_policy(precision_policy)

        with tf.distribute.get_strategy().scope():
            model = get_ffn_model(input_size=input_size, output_size=output_size, hidden_size=8)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            optimizer = GradientAccumulator(optimizer)

            # This is not necessary because the optimizer will be wrapped by Keras see
            # https://github.com/tensorflow/tensorflow/blob/e2af7f7927655e1d0b048bed05afa5e5be8c1f9f/tensorflow/python/keras/engine/training.py#L593

            # if precision_policy.startswith('mixed'):
            #     print(f'Using LossScaleOptimizer for precision-policy "{precision_policy}"')
            #     optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch // train_batch_size,
            validation_data=valid_data,
            validation_steps=10
        )

        loss, accuracy = model.evaluate(valid_data)

        print(f'Evaluation')
        print(f'  - Loss:     {loss:.4f}')
        print(f'  - Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
