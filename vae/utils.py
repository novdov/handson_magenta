import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import magenta.music as mm
import tensorflow.compat.v1 as tf
from logzero import logger
from magenta.models.music_vae import TrainedModel, configs
from magenta.music.protobuf.music_pb2 import NoteSequence

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


def get_model(name: str, batch_size: int = 8):
    checkpoint = f"{name}.tar"
    download_checkpoint("music_vae", checkpoint, "bundles")
    return TrainedModel(
        configs.CONFIG_MAP[name.split(".")[0] if "." in name else name],
        batch_size=batch_size,
        checkpoint_dir_or_path=str(Path("bundles").joinpath(checkpoint)),
    )


def save_midi(
    sequences: Union[NoteSequence, List[NoteSequence]],
    output_dir: Union[str, Path],
    prefix: str = "sequence"
):
    output_dir = Path(output_dir).joinpath("sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(sequences, list):
        sequences = [sequences]

    for idx, sequence in enumerate(sequences):
        date = time.strftime("%Y-%m-%d_%H%M%S")
        filename = f"{prefix}_{idx:02d}_{date}.mid"
        path = str(output_dir.joinpath(filename))
        mm.midi_io.note_sequence_to_midi_file(sequence, path)
        logger.info(f"Generated midi file: {path}")
