import argparse
from pathlib import Path

import magenta.music as mm
from magenta.music import DEFAULT_STEPS_PER_BAR

from vae import core


def get_parser(_=None):
    parser = argparse.ArgumentParser("vae_interpolation")
    parser.add_argument("--model_name", type=str, default="cat-drums_2bar_small.hikl")
    parser.add_argument("--input_midi_1", type=str)
    parser.add_argument("--input_midi_2", type=str)
    parser.add_argument("--num_bar_per_sample", type=int, default=2)
    parser.add_argument("--num_steps_per_bar", type=int, default=DEFAULT_STEPS_PER_BAR)
    parser.add_argument("--num_output", type=int, default=6)
    parser.add_argument("--output_root_dir", type=str)
    return parser


if __name__ == "__main__":

    def _expanduser(_path):
        return str(Path(_path).expanduser())

    known_args, _ = get_parser().parse_known_args()
    output_root_dir = Path(known_args.output_root_dir).expanduser()
    if not output_root_dir.exists():
        output_root_dir.mkdir(parents=True)

    total_bars = known_args.num_output * known_args.num_bar_per_sample
    num_steps_per_sample = known_args.num_bar_per_sample * known_args.num_steps_per_bar
    core.interpolate(
        known_args.model_name,
        start_sequence=mm.midi_file_to_note_sequence(
            _expanduser(known_args.input_midi_1)
        ),
        end_sequence=mm.midi_file_to_note_sequence(
            _expanduser(known_args.input_midi_2)
        ),
        num_steps_per_sample=num_steps_per_sample,
        num_output=known_args.num_output,
        total_bars=total_bars,
        output_dir=output_root_dir,
    )
