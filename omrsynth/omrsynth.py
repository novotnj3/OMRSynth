"""
This module combines individual layers into the resulting synthetic image.
"""

import util.blendops as blendops
import util.imops as imops

import util.general

import util.io as io
import logging
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
import os.path


def combine(sheet_mask, background, fg_struct=1.0, sat_scale=0.45,
            hue_shift=0.0, lev_min=0.15, exp=1.0, out_part=0.3):
    """
    Experimental combination of a sheet image layer with a background layer.
    :param sheet_mask:
    :param background:
    :param fg_struct: How much the background texture affects the foreground
    ink (0.0 without texture, 1.0 fully affected by the texture)
    :param sat_scale: Level of foreground saturation (0.0 desaturated, 1.0 fully
    saturated)
    :param hue_shift: Shift in foreground hue (0.0 no shift)
    :param lev_min: Affects ink density (0.0 fully dense, 1.0 invisible)
    :param exp: Affects "darkness" of the ink (1.0 normal, < 1.0 lighter,
    > 1.0 darker)
    :param out_part: Intensity of ink dissolving on the foreground outline
    (0.0 no outline - sharp borders, 0.3-0.5 fair borders, 1.0 sharp and bold)
    :return:
    """

    morph_selem = np.ones((3, 3))
    sheet_mask_eroded = morph.erosion(sheet_mask, selem=morph_selem)

    # Enable blending with RGB by converting grayscale to 3 channels
    sheet_rgb = util.general.repeat_to_3_channels(1 - sheet_mask).astype(np.bool)
    sheet_eroded_rgb = util.general.repeat_to_3_channels(sheet_mask_eroded)

    # Perform experimental blending simulating printed music sheet
    added = blendops.addition(background, sheet_eroded_rgb)
    white_bal = imops.white_balance(added)
    white_bal = blendops.normal(white_bal, sheet_eroded_rgb, fg_struct)
    sat_scaled = imops.hsv_scale(white_bal, sat_scale)
    levels = util.general.linear_transform(sat_scaled, y_min=lev_min)
    masked = blendops.addition(sheet_eroded_rgb, levels)

    overlay_sharp = blendops.multiply(masked, background) ** exp
    overlay_sharp = imops.hsv_scale(overlay_sharp, hue_shift=hue_shift)

    # Outline:
    morph_mask = 1 - sheet_mask
    dilated = morph.binary_dilation(morph_mask)
    outer_outline = np.logical_xor(dilated, morph_mask)
    outer_outline = util.general.repeat_to_3_channels(outer_outline)

    overlay_final = background.copy()
    overlay_final[sheet_rgb] = overlay_sharp[sheet_rgb]

    overlay_final[outer_outline] *= overlay_sharp[outer_outline]
    u, v = out_part, 1 - out_part
    overlay_final[outer_outline] = (u * overlay_final[outer_outline] ** exp
                                    + v * background[outer_outline])

    return overlay_final


def main():
    import util.io as io
    sheet = io.load_binary_float('/media/jirka/DATA/MFF/OMR Data/PXCNN/dalitz/schumann.png')
    bcg = io.load_float('/media/jirka/DATA/MFF/OMR Data/PXCNN/backgrounds/schumann_fibich_poem.png')

    sr, sc = 1870, 70
    sheet = sheet[sr:sr+580, sc:sc+480]
    bcg = bcg[sr:sr+580, sc:sc+480]

    comb = combine(sheet, bcg, fg_struct=1.0, sat_scale=0.8, hue_shift=-0.25,
                   lev_min=0.4, exp=1.6, out_part=0.24)

    io.save_image('/media/jirka/DATA/Disk Google/MFF/Slides/Pixel CNN/img/synthcomp2.jpg', comb)
    plt.imshow(comb, interpolation='nearest')
    plt.show()


def pixel_cnn_experiment():
    params = {'dvorak': (0.8, 0.4, -0.05, 0.1, 1.2, 0.25),
              'empty': (1.0, 0.65, 0.0, 0.2, 0.75, 0.3),
              'fibich_poem': (0.9, 0.3, -0.15, 0.3, 1.25, 0.15),
              'fibich_vodnik': (0.7, 0.45, 0.0, 0.25, 1.1, 0.3),
              'gershwin': (0.6, 0.5, 0.0, 0.1, 1.3, 0.4),
              'satie': (0.5, 0.45, -0.1, 0.35, 1.1, 0.15)}

    path_bcgs = '/media/jirka/DATA/MFF/OMR Data/PXCNN/backgrounds'
    path_sheets = '/media/jirka/DATA/MFF/OMR Data/PXCNN/dalitz'
    path_output = '/media/jirka/DATA/MFF/OMR Data/PXCNN/combined'

    backgrounds = ['dvorak', 'empty', 'fibich_poem', 'fibich_vodnik',
                   'gershwin', 'satie']

    sheets = ['bach', 'bellinzani', 'brahms02', 'bruckner01', 'buxtehude',
                 'carcassi01', 'dalitz03', 'diabelli', 'mahler', 'pmw01',
                 'pmw03', 'pmw04', 'rameau', 'schumann', 'tye', 'victoria09',
                 'wagner', 'williams']

    counter = 0
    for bcg in backgrounds:
        for sheet in sheets:
            filename_bcg = '{0}_{1}.png'.format(sheet, bcg)
            filename_gray = '{0}_{1}_gray.png'.format(sheet, bcg)
            bcg_in = os.path.join(path_bcgs, filename_bcg)

            percent = 100.0 * counter / (len(backgrounds) * len(sheets))
            message = '{0:.2f} %: combining with background {1}'
            logging.info(message.format(percent, bcg_in))

            filename_sheet = '{0}.png'.format(sheet)
            sheet_in = os.path.join(path_sheets, filename_sheet)

            filename_out = filename_bcg
            file_out = os.path.join(path_output, filename_out)
            file_out_gray = os.path.join(path_output, filename_gray)

            sheet_im = io.load_binary_float(sheet_in)
            bcg_im = io.load_float(bcg_in)

            settings = params[bcg]
            combined = combine(sheet_im, bcg_im, *settings)

            io.save_image(file_out, combined)
            io.save_image_gray(file_out_gray, combined)

            counter += 1


def generate_groundtruth():
    path_sheets = '/media/jirka/DATA/MFF/OMR Data/PXCNN/dalitz'
    path_output = '/media/jirka/DATA/MFF/OMR Data/PXCNN/groundtruth'

    sheets = ['bach', 'bellinzani', 'brahms02', 'bruckner01', 'buxtehude',
                 'carcassi01', 'dalitz03', 'diabelli', 'mahler', 'pmw01',
                 'pmw03', 'pmw04', 'rameau', 'schumann', 'tye', 'victoria09',
                 'wagner', 'williams']

    counter = 0
    for sheet in sheets:
        percent = 100.0 * counter / len(sheets)
        message = '{0:.2f} %: generating groundtruth for {1}'
        logging.info(message.format(percent, sheet))

        filename_sheet = '{0}.png'.format(sheet)
        sheet_in = os.path.join(path_sheets, filename_sheet)

        filename_fg = '{0}-nostaff.png'.format(sheet)
        fg_in = os.path.join(path_sheets, filename_fg)

        full = 255 - io.load_uint8(sheet_in)
        out_im = 255 - io.load_uint8(fg_in)

        # Add staffline pixels
        stafflines = np.logical_xor(full, out_im)
        out_im[stafflines] = 128

        file_out = os.path.join(path_output, filename_sheet)
        io.save_image(file_out, out_im)

        counter += 1


if __name__ == '__main__':
    import json
    import logging.config
    json_file = open('./logging.json')
    config = json.load(json_file)
    logging.config.dictConfig(config)
    logging.info('Welcome to OMR Synthesizer')

    #generate_groundtruth()
    main()
