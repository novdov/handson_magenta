from pathlib import Path
from typing import List, Union

from magenta.music.protobuf.music_pb2 import NoteSequence

from . import utils


def sample(
    model_name: str,
    num_steps_per_sample: int,
    output_dir: Union[str, Path],
    **sample_kwargs
) -> List[NoteSequence]:
    model = utils.get_model(model_name)
    sample_sequences = model.sample(n=2, length=num_steps_per_sample, **sample_kwargs)
    utils.save_midi(sample_sequences, output_dir)
    return sample_sequences
