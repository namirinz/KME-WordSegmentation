from typing import List

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.layers as layers
from kmeseg.utils.config import CHAR_INDICES_SIZE, LOOK_BACK


def create_model():
    model = tf.keras.Sequential(
        [
            layers.Input((LOOK_BACK,)),
            layers.Embedding(CHAR_INDICES_SIZE, 64, input_length=LOOK_BACK),
            layers.Bidirectional(
                layers.GRU(32, return_sequences=False), merge_mode="sum"
            ),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def compile_model(
    model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, scaled_lr: float
):
    """Compile the model.

    Args:
        model (tf.keras.Model): [description]
        optimizer (tf.keras.optimizer.Optimizer): [description]
        learning_rate ([type]): [description]
    """
    opt = f"tf.keras.optimizers.{optimizer}(learning_rate={scaled_lr})"
    opt = eval(opt)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )


def create_callbacks(
    scaled_lr: float,
    modelpath: str,
    early_stop_cb_patience: int,
    reduce_lr_cb_patience: int,
    use_tensorboard: bool,
) -> List:
    """Creating the list of callbacks for training.

    Args:
        scaled_lr (float): Learning rate scaled by the number of workers.
        modelpath (str): Path to save model.
        early_stop_cb_patience (int): Number of patience for early stopping callback.
        reduce_lr_cb_patience (int): Number of patience for reduce learning rate callback.
        use_tensorboard (bool): If True, use tensorboard callback.

    Returns:
        List: List of callbacks.
    """
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=scaled_lr, warmup_epochs=5, verbose=1
        ),
    ]
    if hvd.rank() == 0:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=modelpath,
            include_optimizer=False,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            verbose=1,
        )
        callbacks.append(checkpoint_cb)

    if early_stop_cb_patience != -1:
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop_cb)

    if reduce_lr_cb_patience != -1:
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3, verbose=1
        )
        callbacks.append(reduce_lr_cb)

    if use_tensorboard:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir="./models/logs", update_freq=1000
        )
        callbacks.append(tensorboard_cb)

    return callbacks
