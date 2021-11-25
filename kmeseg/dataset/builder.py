import numpy as np
import pandas as pd
import tensorflow as tf
from kmeseg.utils.config import CHAR_INDICES, LOOK_BACK
from tensorflow.data import AUTOTUNE, Dataset


def create_dataset(text: str) -> np.ndarray:
    """
    take text with label (text that being defined where to cut ('|'))
    and encode text and make label
    return preprocessed text & preprocessed label
    """
    samples, labels = [], []
    text = "|" + text
    data = [CHAR_INDICES["<pad>"]] * LOOK_BACK
    for i in range(1, len(text)):
        current_char = text[i]
        before_char = text[i - 1]

        if current_char == "|":
            continue
        data = data[1:] + [CHAR_INDICES[current_char]]  # X data

        target = 1 if before_char == "|" else 0  # y data
        samples.append(data)
        labels.append(target)

    return np.array(samples), np.array(labels)


def prepare_dataset(arr_label):

    return "|".join(arr_label)


def create_tf_dataset(
    samples: np.ndarray,
    label: np.ndarray,
    buffer_size: int,
    batch_size: int,
    use_cache: bool,
) -> tf.data.Dataset:
    dataset = Dataset.from_tensor_slices((samples, label))
    dataset = dataset.shuffle(buffer_size).batch(batch_size)

    if use_cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def build_dataset(
    df: pd.DataFrame,
    label_col: str,
    n_samples: int,
    buffer_size: int = 1000,
    batch_size: int = 32,
    use_cache: bool = True,
) -> tf.data.Dataset:
    iupaccut = df[label_col].iloc[:n_samples].values
    
    print(f"Selecting {len(iupaccut)} samples.")

    iupaccut = prepare_dataset(iupaccut)

    samples, label = create_dataset(iupaccut)

    dataset = create_tf_dataset(
        samples, label, buffer_size, batch_size, use_cache
    )

    return dataset
