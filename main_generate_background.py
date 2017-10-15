"""
Script for generating background textures using OMRSynth.
"""

import argparse

from omrsynth.paper.texturesynth import TextureSynth


def parse_args():
    parser = argparse.ArgumentParser('OMRSynth background generator')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to the texture sample image file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the generated background image.')
    parser.add_argument('--width', type=int, required=True,
                        help='Width of the generated image.')
    parser.add_argument('--height', type=int, required=True,
                        help='Height of the generated image.')

    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Random seed for the generator.')
    parser.add_argument('--impurities', type=float, required=False, default=1,
                        help='Amount of impurities. A float value from 0 to 1.'
                             ' One for roughly the same amount of impurities '
                             'as in the input sample, zero for no impurities.')

    return parser.parse_args()


def main():
    args = parse_args()

    synthesizer = TextureSynth().random_seed(args.seed).impurities(
        args.impurities)
    out_shape = (args.height, args.width)
    synthesizer.generate_texture(args.input, args.output, out_shape)


if __name__ == '__main__':
    main()
