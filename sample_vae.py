import argparse
from pathlib import Path

from magenta.music import DEFAULT_STEPS_PER_BAR

from vae import sampling


def get_parser(_=None):
    parser = argparse.ArgumentParser("MusicVAE/GrooVAE")
    parser.add_argument("--model_name", type=str, default="cat-drums_2bar_small.lokl")
    parser.add_argument("--num_bar_per_sample", type=int, default=2)
    parser.add_argument("--num_steps_per_bar", type=int, default=DEFAULT_STEPS_PER_BAR)
    parser.add_argument("--output_dir", type=str)
    return parser


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    output_dir = Path(known_args.output_dir).expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    num_steps_per_sample = known_args.num_bar_per_sample * known_args.num_steps_per_bar
    sampling.sample(known_args.model_name, num_steps_per_sample, output_dir)
