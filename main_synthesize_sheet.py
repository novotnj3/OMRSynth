"""
Script for synthesizing sheets using OMRSynth.
"""

import argparse

import omrsynth.util.io as io
from omrsynth.ink import basic_print
from omrsynth.omrsynth import combine


def parse_args():
    parser = argparse.ArgumentParser('OMRSynth sheet generator')

    parser.add_argument('--input_sheet', type=str, required=True,
                        help='Path to the binary sheet image file.')
    parser.add_argument('--background', type=str, required=True,
                        help='Path to the background image file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the generated sheet image.')

    # Parameters of basic print simulation
    parser.add_argument('--level', type=float, required=False, default=0.1,
                        help='Level of printed noise degradation.')
    parser.add_argument('--blur', type=int, required=False, default=3,
                        help='Size of the blurring kernel.')

    # Parameters for combining the simulated print with background
    parser.add_argument('--leak', type=float, required=False, default=1.0,
                        help='How much the background roughness affects the '
                             'foreground ink (0.0 no affection, 1.0 fully '
                             'affected by the texture.')
    parser.add_argument('--saturation', type=float, required=False,
                        default=0.45, help='Level of ink saturation (0.0 '
                                           'desaturated, 1.0 fully saturated.')
    parser.add_argument('--hue_shift', type=float, required=False, default=0.0,
                        help='Shift of foreground ink hue (0.0 no shift, '
                             'typically +/- some small value)')
    parser.add_argument('--density', type=float, required=False, default=0.15,
                        help='Affects ink density in a different way than '
                             'saturation (0.0 fully dense, 1.0 invisible).')
    parser.add_argument('--darkness', type=float, required=False, default=1.0,
                        help='Affects "darkness" of the ink (1.0 normal, < 1.0'
                             ' lighter, > 1.0 darker).')
    parser.add_argument('--outline', type=float, required=False, default=0.3,
                        help='Intensity of ink dissolving on the foreground '
                             'outline (0.0 no outline - sharp borders, '
                             '0.3-0.5 fair borders, 1.0 sharp and bolder)')

    return parser.parse_args()


def main():
    args = parse_args()

    binary_sheet = io.load_binary_float(args.input_sheet)
    background = io.load_rgb_float(args.background)
    printed = basic_print(binary_sheet, level=args.level, blur_size=args.blur)
    combined = combine(printed, background, fg_struct=args.leak,
                       sat_scale=args.saturation, hue_shift=args.hue_shift,
                       lev_min=args.density, exp=args.darkness,
                       out_part=args.outline)

    io.save_image(args.output, combined)


if __name__ == '__main__':
    main()
