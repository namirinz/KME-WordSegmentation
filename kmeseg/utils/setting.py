import os
from typing import List

import tensorflow as tf


def setup_path():
    project_dir = os.path.abspath("")
    data_dir = os.path.join(project_dir, "data")

    return project_dir, data_dir


def setup_gpu(gpu_ids: List[str], local_rank: int) -> None:
    print("----- Setting GPU -----")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    gpus = tf.config.experimental.list_physical_devices("GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{gpu.name} initialized.")

    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[local_rank], "GPU"
        )

    print("-----------------------")
