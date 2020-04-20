import time
import urllib.request
from pathlib import Path
from typing import Callable, List, Union

import magenta.music as mm
import tensorflow.compat.v1 as tf
from logzero import logger
from magenta.models.music_vae import TrainedModel, configs
from magenta.music.protobuf.music_pb2 import NoteSequence
from visual_midi import Plotter

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


class MIDIWriter:
    def __init__(
        self, sub_dir_name: str = "sample", time_format: str = "%Y-%m-%d_%H%M%S"
    ):
        self._sub_dir_name = sub_dir_name
        self._time_format = time_format

        self._logging_format = "Generated {file_type} file: {path}"

    def write_midi(
        self,
        sequences: Union[NoteSequence, List[NoteSequence]],
        output_dir: Union[str, Path],
        prefix: str = "sequence",
    ) -> None:
        def _save_midi(sequence, path):
            mm.midi_io.note_sequence_to_midi_file(sequence, path)
            logger.info(self._logging_format.format(file_type="midi", path=path))

        self._write(
            _save_midi,
            output_format="mid",
            sequences=sequences,
            output_dir=output_dir,
            prefix=prefix,
        )

    def write_plot(
        self,
        sequences: Union[NoteSequence, List[NoteSequence]],
        output_dir: Union[str, Path],
        prefix: str = "sequence",
        **plot_kwargs,
    ) -> None:
        def _save_plot(sequence, path):
            midi = mm.midi_io.note_sequence_to_pretty_midi(sequence)
            plotter = Plotter(**plot_kwargs)
            plotter.save(midi, path)
            logger.info(self._logging_format.format(file_type="plot", path=path))

        self._write(
            _save_plot,
            output_format="html",
            sequences=sequences,
            output_dir=output_dir,
            prefix=prefix,
        )

    def _write(
        self,
        write_fn: Callable[[List[NoteSequence], str], None],
        output_format: str,
        sequences: Union[NoteSequence, List[NoteSequence]],
        output_dir: Union[str, Path],
        prefix: str = "sequence",
    ) -> None:
        output_dir = Path(output_dir).joinpath(self._sub_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(sequences, list):
            sequences = [sequences]

        for idx, sequence in enumerate(sequences):
            filename = (
                f"{prefix}_{idx:02d}_{time.strftime(self._time_format)}.{output_format}"
            )
            path = str(output_dir.joinpath(filename))
            write_fn(sequence, path)
