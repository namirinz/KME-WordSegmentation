import tensorflow.keras as tfk
import tensorflow.keras.layers as layers
from kmeseg.utils.config import CHAR_INDICES_SIZE, LOOK_BACK


def create_model():
    model = tfk.Sequential(
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


def compile_model(model, optimizer, learning_rate):
    opt = f"tfk.optimizers.{optimizer}(learning_rate={learning_rate})"
    model.compile(
        optimizer=eval(opt),
        loss=tfk.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )


def create_callbacks(
    modelpath, early_stop_cb_patience, reduce_lr_cb_patience, use_tensorboard
):
    checkpoint_cb = tfk.callbacks.ModelCheckpoint(
        filepath=modelpath,
        include_optimizer=True,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        verbose=0,
    )
    callback_list = [checkpoint_cb]

    if early_stop_cb_patience != -1:
        early_stop_cb = tfk.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )
        callback_list.append(early_stop_cb)

    if reduce_lr_cb_patience != -1:
        reduce_lr_cb = tfk.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3, verbose=0
        )
        callback_list.append(reduce_lr_cb)

    if use_tensorboard:
        tensorboard_cb = tfk.callbacks.TensorBoard(
            log_dir="./models/logs", update_freq=1000
        )
        callback_list.append(tensorboard_cb)

    return callback_list
