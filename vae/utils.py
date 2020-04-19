import urllib.request
from pathlib import Path

import tensorflow.compat.v1 as tf
from logzero import logger

MODEL_DOWNLOAD_ROOT = "https://storage.googleapis.com/magentadata/models/"
MODEL_DOWNLOAD_FMT = MODEL_DOWNLOAD_ROOT + "{model_name}/checkpoints/{checkpoint_name}"


def download_checkpoint(model_name: str, checkpoint_name: str, target_dir: str) -> None:
    tf.io.gfile.makedirs(target_dir)
    checkpoint_target = Path(target_dir).joinpath(checkpoint_name)
    if not checkpoint_target.exists():
        logger.info(
            f"Downloading {model_name}:{checkpoint_name.split('.')[0]} to {target_dir}"
        )
        response = urllib.request.urlopen(
            MODEL_DOWNLOAD_FMT.format(
                model_name=model_name, checkpoint_name=checkpoint_name
            )
        )
        data = response.read()
        with open(checkpoint_target, "wb") as f_out:
            f_out.write(data)
