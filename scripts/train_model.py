import os
from argparse import ArgumentParser
from typing import List, Tuple

import horovod.tensorflow.keras as hvd
import pandas as pd
import tensorflow as tf
from kmeseg.core.model import compile_model, create_callbacks, create_model
from kmeseg.dataset.builder import build_dataset
from kmeseg.utils.setting import setup_gpu, setup_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
PROJECT_PATH, DATA_PATH = setup_path()

hvd.init()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=["0"],
        help="Select id you want to use.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="Number of sample to use.",
    )
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Select batch size.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs."
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Model optimizer."
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Starting Learning Rate.",
    )
    parser.add_argument("--model-name", type=str, default="kmeseg_model.h5")
    parser.add_argument(
        "--escb-patience", type=int, default=5, help="Early Stopping patience."
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum difference loss between epoch to be stop.",
    )
    parser.add_argument(
        "--rlcb-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--use-tensorboard",
        type=bool,
        default=False,
        help="Whether to using Tensorboard.",
    )
    parser.add_argument(
        "--use-cache", type=bool, default=False, help="Whether to use cache."
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=True,
        help="Whether to visualize the training loss.",
    )

    args = parser.parse_args()

    return args


def get_dataset(
    buffer_size: int,
    batch_size: int,
    n_samples: int, use_cache: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Loading dataset from data folder and build the dataset.

    Args:
        buffer_size (int): Length of sample to shuffle.
        batch_size (int): Number of batch size.
        use_cache (bool): Storing batch data to RAM (Consider the RAM size before using).

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Dataset to be use
    """
    print("---- Building the dataset ----")
    dataframe_path = os.path.join(DATA_PATH, "processed")
    df_train = pd.read_csv(f"{dataframe_path}/train/df_train.csv")
    df_val = pd.read_csv(f"{dataframe_path}/val/df_val.csv")

    dataset = build_dataset(
        df=df_train,
        label_col="label",
        n_samples=n_samples,
        buffer_size=buffer_size,
        batch_size=batch_size,
        use_cache=use_cache,
    )

    validation_dataset = build_dataset(
        df=df_val,
        label_col="label",
        n_samples=n_samples,
        buffer_size=buffer_size,
        batch_size=batch_size,
        use_cache=use_cache,
    )
    print("---- Getting Dataset Successful ----")

    return dataset, validation_dataset


def train(
    dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int,
    batch_size: int,
    callbacks: List,
    verbose: int,
):
    """Training the model.

    Args:
        dataset (tf.data.Dataset): Training Dataset contain Sentences and label.
        validation_dataset (tf.data.Dataset): Validation Dataset.
        model (tfk.Model): Word Segmentation Model.
        epochs (int): Number of training for the model.
        batch_size (int): Number of Batch size.
        callbacks (List): List of Tensorflow Callbacks to use.
    """
    print("---- Training Model ----")
    model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    print("---- Training Model Successful ----")


def evaluate(dataset: tf.data.Dataset, model: tf.keras.Model, batch_size: int):
    """Evaluating Model Performance.

    Args:
        dataset (tf.data.Dataset): Testing Dataset
        model (tfk.Model): Model to be evaluate.
        batch_size (int): Number of Batch size.
    """
    print("---- Evaluating Model ----")
    model.evaluate(dataset, batch_size=batch_size)
    print("---- Evaluating Model Successful ----")


def main():
    args = parse_args()
    BATCH_SIZE = args.batch_size

    setup_gpu(args.gpu_ids, hvd.local_rank())

    if len(args.gpu_ids) > 1:
        model = create_model()
        BATCH_SIZE = BATCH_SIZE
        print(f"Number of devices: {hvd.size()}")
    else:
        print(f"Number of devices: 1")
        model = create_model()

    scaled_lr = args.learning_rate * hvd.size()
    compile_model(model, optimizer=args.optimizer, scaled_lr=scaled_lr)

    callback_list = create_callbacks(
        modelpath=f"{PROJECT_PATH}/models/save_model/{args.model_name}",
        early_stop_cb_patience=args.escb_patience,
        reduce_lr_cb_patience=args.rlcb_patience,
        use_tensorboard=args.use_tensorboard,
        scaled_lr=scaled_lr,
    )

    dataset, validation_dataset = get_dataset(
        buffer_size=args.buffer_size,
        batch_size=BATCH_SIZE,
        n_samples=args.n_samples,
        use_cache=args.use_cache,
    )

    verbose = 1 if hvd.rank() == 0 else 0

    train(
        dataset=dataset,
        validation_dataset=validation_dataset,
        model=model,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
