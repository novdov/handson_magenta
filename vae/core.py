from pathlib import Path
from typing import List, Union

import magenta.music as mm
from magenta.music.protobuf.music_pb2 import NoteSequence

from . import utils


def sample(
    model_name: str,
    num_steps_per_sample: int,
    output_dir: Union[str, Path],
    **sample_kwargs,
) -> List[NoteSequence]:
    model = utils.get_model(model_name)
    sample_sequences = model.sample(n=2, length=num_steps_per_sample, **sample_kwargs)

    midi_writer = utils.MIDIWriter()
    midi_writer.write_midi(sample_sequences, output_dir)
    midi_writer.write_plot(sample_sequences, output_dir)
    return sample_sequences


def interpolate(
    model_name: str,
    start_sequence: NoteSequence,
    end_sequence: NoteSequence,
    num_steps_per_sample: int,
    num_output: int,
    total_bars: int,
    output_dir: Union[str, Path],
) -> NoteSequence:
    model = utils.get_model(model_name)
    interpolate_sequences = model.interpolate(
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        num_steps=num_output,
        length=num_steps_per_sample,
    )

    midi_writer = utils.MIDIWriter()
    midi_writer.write_midi(interpolate_sequences, output_dir, prefix="interpolate")
    midi_writer.write_plot(interpolate_sequences, output_dir, prefix="interpolate")

    interpolate_sequence = mm.concatenate_sequences(
        interpolate_sequences, sequence_durations=[4] * num_output
    )
    midi_writer.write_midi(interpolate_sequence, output_dir, prefix="merge")
    midi_writer.write_plot(
        interpolate_sequence,
        output_dir,
        prefix="merge",
        plot_max_length_bar=total_bars,
        bar_fill_alphas=[0.5, 0.5, 0.05, 0.05],
    )

    return interpolate_sequence
