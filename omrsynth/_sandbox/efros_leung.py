import numpy as np
import skimage.morphology as skmorph
import skimage.io
import skimage


def synthesize_texture(in_image, out_height, out_width, window_size,
                       random_seed=1):
    """
    Synthesizes a texture given an input image and desired image sizes using
    the Efros-Leung algorithm for texture synthesis by non-parametric sampling.

    See the original paper:
    https://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf
    :param in_image: An input image containing a sample of the texture.
    :param out_height: Desired height of the synthesized image.
    :param out_width: Desired width of the synthesized image.
    :param window_size: Size of the square window used to scan the input image
    to find the best match. The size of the window should be on the scale
    of the biggest regular feature.
    :param random_seed: Random seed used to initialize the random generator.
    :return: An out_height x out_width image with the synthesized texture.
    """

    # Initialize numpy's random:
    np.random.seed(random_seed)

    # Window size is assumed to be odd. Add one if the size is even, because
    # larger windows generally leads to better results:
    if window_size % 2 == 0:
        window_size += 1

    # Create an extra dimension if the input image is in grayscale:
    if len(in_image.shape) < 3:
        in_image = in_image[:, :, np.newaxis]

    # Prepare the output image and place the initial seed pixels:
    in_height, in_width, in_channels = in_image.shape
    out_image = np.zeros((out_height, out_width, in_channels))
    mask_matrix = _initial_seed(in_image, out_image)

    # Pad the arrays for easier neighborhood manipulations by half-window
    # sized borders filled with zeros:
    half_window = window_size // 2
    half_w_tuple = (half_window, half_window)

    pad_w_out = (half_w_tuple, half_w_tuple, (0, 0))
    out_image_pad = np.pad(array=out_image,
                           pad_width=pad_w_out,
                           mode='constant',
                           constant_values=0)

    pad_w_mask = (half_w_tuple, half_w_tuple)
    mask_matrix_pad = np.pad(array=mask_matrix,
                             pad_width=pad_w_mask,
                             mode='constant',
                             constant_values=0)

    # Count the number of times the window can slide by one pixel in vertical
    # (row_extent) and horizontal (col_extent) directions.
    row_extent = in_height - window_size + 1
    col_extent = in_width - window_size + 1

    # Allocate the candidates - each candidate patch is represented as
    # a column vector (reshaped from the squared windows).
    num_of_candidates = row_extent * col_extent
    candidates_shape = (window_size ** 2, num_of_candidates, in_channels)
    candidates = np.zeros(shape=candidates_shape, dtype=float)

    # Fill the candidates for each channel separately:
    for channel in range(in_channels):
        ch_candidates = _get_candidates(in_image[:, :, channel], window_size)
        candidates[:, :, channel] = ch_candidates

    # Reshape the candidates to be a two-dimensional matrix by stacking:
    stacked_candidates = _stack_channels(candidates)

    # Prepare a gaussian matrix, flatten it, repeat its elements for each
    # channel and make it a column vector:
    sigma = 6.4
    gaussian = _gaussian_kernel_2d(window_size, window_size / sigma)
    gaussian_vec = np.repeat(gaussian.ravel(), in_channels)[:, np.newaxis]

    # Algorithm parameters:
    error_threshold = 0.1
    max_error_threshold = 0.3

    # Repeat while all the output pixels are not filled:
    while not mask_matrix.all():
        # Have not found anything yet:
        found_match = False

        # Loop pixels to be filled in this step:
        for pix_row, pix_col in _next_unfilled_pixels(mask_matrix):
            neighborhood, mask = _get_neighborhood(el_row=pix_row,
                                                   el_col=pix_col,
                                                   window_size=window_size,
                                                   padded_image=out_image_pad,
                                                   padded_mask=mask_matrix_pad)

            # Reshape the neighborhood into a column vector:
            neighborhood_vec = neighborhood.reshape((-1, 1))

            # Create a matrix where the neighborhood vector is repeated
            # horizontally number-of-candidates times:
            neighborhood_rep = np.tile(neighborhood_vec,
                                       (1, num_of_candidates))

            # Reshape the mask into a column vector analogically:
            mask_vec = np.repeat(mask.ravel(), in_channels)[:, np.newaxis]

            # Find the sum of the valid Gaussian elements:
            gaussian_mask_vec = gaussian_vec * mask_vec
            weight = gaussian_mask_vec.sum()

            # Create a row vector with valid normalized gaussian elements:
            gaussian_mask = (gaussian_mask_vec / weight).transpose()

            # Compute gaussian weighted distances between the current
            # neighborhood and all the candidate patches.
            sqr_errors = (stacked_candidates - neighborhood_rep) ** 2
            distances = np.dot(gaussian_mask, sqr_errors).ravel()

            # Find the indices where distances are less than the threshold:
            min_threshold = distances.min() * (1 + error_threshold)
            min_positions = np.where(distances <= min_threshold)[0]

            # Randomly select a candidate patch from the suited ones:
            selected_position = np.random.choice(min_positions)
            selected_error = distances[selected_position]

            if selected_error < max_error_threshold:
                # Compute row and column indices and shift them so they match
                # the central point of the selected candidate patch:
                sel_row = (selected_position // col_extent) + half_window
                sel_col = (selected_position % col_extent) + half_window

                # Copy the matched pixel and record it was found:
                out_image[pix_row, pix_col, :] = in_image[sel_row, sel_col, :]
                mask_matrix[pix_row, pix_col] = 1.0
                found_match = True

        if found_match:
            # Update the interior part of padded images to reflect updates:
            pad = half_window
            out_image_pad[pad:-pad, pad:-pad, :] = out_image
            mask_matrix_pad[pad:-pad, pad:-pad] = mask_matrix
        else:
            # If no pixel has been filled in this step, increase the error
            # threshold to ensure convergence of the algorithm:
            max_error_threshold *= 1.1

    return out_image


def _initial_seed(in_image, out_image, seed_size=5):
    """
    Places a random patch taken from the input into the centre of the output.
    :param in_image: The original image used for patch sampling.
    :param out_image: The output image, where the patch will be placed.
    :param seed_size: Size of the initial square seed patch.
    :return: A boolean mask determining where the pixels are already filled.
    """

    # Ensure odd seed size:
    if seed_size % 2 == 0:
        seed_size += 1

    # Get the input dimensions:
    in_height, in_width = in_image.shape[0:2]
    out_height, out_width = out_image.shape[0:2]

    # Compute right/bottom margin:
    margin = seed_size - 1

    # Extract a random (seed_size x seed_size) patch from the input image:
    rand_row = np.random.randint(0, in_height - margin)
    rand_col = np.random.randint(0, in_width - margin)
    seed_patch = in_image[rand_row:(rand_row + seed_size),
                          rand_col:(rand_col + seed_size), :]

    # Place the patch in the centre of the output image:
    cent_row = out_height // 2
    cent_col = out_width // 2

    # For even output height shift central row one pixel to the up:
    if out_height % 2 == 0:
        cent_row -= 1

    # For even output width shift central column one pixel to the left:
    if out_width % 2 == 0:
        cent_col -= 1

    half_seed_size = seed_size // 2

    # Compute min and max row and column indices:
    r_min = cent_row - half_seed_size
    r_max = r_min + seed_size
    c_min = cent_col - half_seed_size
    c_max = c_min + seed_size

    # Copy the seed patch into the output image:
    out_image[r_min:r_max, c_min:c_max, :] = seed_patch

    # Create a mask boolean map, where already filled pixels are True
    mask_matrix = np.zeros((out_height, out_width), dtype=float)
    mask_matrix[r_min:r_max, c_min:c_max] = 1.0

    return mask_matrix


def _next_unfilled_pixels(mask_matrix):
    """
    Returns a list of unfilled pixels to be filled in the next step.
    :param mask_matrix: A boolean matrix representing already filled pixels.
    :return: A list of (row, column) indices of the pixels to be filled
    in the next step, i.e. unfilled pixels neighboring the filled ones.
    """

    # Choose 3x3 square structuring element:
    element = skmorph.square(3)

    # Dilate already filled pixel map and subtract it from the result
    # to get the next "onion layer":
    dilated_mask = skmorph.binary_dilation(mask_matrix, element)
    diff_mask = dilated_mask - mask_matrix

    # Stack the result in column-wise manner and return pixel indices:
    return np.column_stack(np.where(diff_mask == 1))


def _get_neighborhood(el_row, el_col, window_size, padded_image, padded_mask):
    """
    Returns the neighborhood of a specified pixel and its filling mask.
    :param el_row: The row of the pixel at the center of the neighborhood.
    :param el_col: The column of the pixel at the center of the
    neighborhood.
    :param window_size: Size of the neighborhood.
    :param padded_image: The padded output image (by a half window size on all
    sides).
    :param padded_mask: The padded "already filled" boolean pixels mask.
    :return: A tuple (neighborhood, mask) containing the square regions of
    (window_size x windows_size) of the neighborhood pixels and their filling
    mask.
    """

    half_windows_size = window_size // 2

    # Add a padding size to get correct indices to the padded images:
    el_row += half_windows_size
    el_col += half_windows_size

    # Compute the row and column neighborhood indices:
    r_min = el_row - half_windows_size
    r_max = r_min + window_size
    c_min = el_col - half_windows_size
    c_max = c_min + window_size

    # Return the pixel neighborhood and its mask:
    neighborhood = padded_image[r_min:r_max, c_min:c_max, :]
    mask = padded_mask[r_min:r_max, c_min:c_max]
    return neighborhood, mask


def _get_candidates(matrix, block_size):
    """
    Returns list of all possible blocks (candidates) created by sliding
    a (block_size x block_size) window across the input matrix. Sliding is
    performed in the left-to-right and up-to-down manner always shifted by
    one element.
    :param matrix: The input two-dimensional matrix from which to choose
    candidates.
    :param block_size: Size of the square sliding window.
    :return: Matrix of candidates, i.e. a matrix with block_size^2 rows and
    total number of candidates columns. Each column corresponds to the
    individual candidate and its rows are filled with corresponding elements
    (square blocks are flattened in row-wise manner).
    """

    # Compute parameters:
    h, w = matrix.shape[0:2]
    col_extent = w - block_size + 1
    row_extent = h - block_size + 1

    # Get starting block indices:
    start_idx = (w * np.arange(block_size)[:, np.newaxis]
                 + np.arange(block_size))

    # Get offset indices across the height and width of input array
    offset_idx = (w * np.arange(row_extent)[:, np.newaxis]
                  + np.arange(col_extent))

    # Get indices of individual blocks flattened in row-wise manner:
    block_indices = start_idx.ravel()[:, np.newaxis] + offset_idx.ravel()
    return np.take(matrix, block_indices)


def _stack_channels(input_array):
    """
    Transforms a three-dimensional array into two-dimensional by stacking
    its channel values (along third dimension) vertically.

    For example, let the input X be (2 x 4 x 3)-array. The output Y is then
    (6 x 4)-array (because 6 = 2 * 3) and following relationships are valid:
    Y[0:2, :] = X[0, :, 0:2]
    Y[3:5, :] = X[1, :, 0:2]
    :param input_array: The input three-dimensional array.
    :return: The transformed two-dimensional array.
    """

    swapped_axes = input_array.swapaxes(1, 2)
    return swapped_axes.reshape((-1, input_array.shape[1],))


def _gaussian_kernel_2d(size=3, sigma=1.0):
    """
    Creates a rotationally symmetric square Gaussian kernel of a given size
    and a standard deviation.
    :param size: Desired size of the kernel.
    :param sigma: Positive standard deviation of the Gaussian.
    :return: (size x size) Gaussian kernel with specified standard deviation.
    """

    s = (size - 1.) / 2.
    y, x = np.ogrid[-s:(s + 1), -s:(s + 1)]

    # Compute the Gaussian and filter too small values:
    gauss = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0

    # Normalize:
    sum_gauss = gauss.sum()
    if sum_gauss != 0:
        gauss /= sum_gauss

    return gauss


def main():
    file_in = '../../imgs/textures/samples/classical1.png'
    file_out = '../../imgs/textures/synthesized/efros-leung.png'

    out_width = 256
    out_height = 256
    window_size = 7
    random_seed = 0

    input_patch = skimage.io.imread(file_in)
    input_patch = skimage.img_as_float(input_patch)
    texture = synthesize_texture(input_patch,
                                 out_height,
                                 out_width,
                                 window_size,
                                 random_seed)
    if texture.shape[2] == 1:
        texture = texture.squeeze()

    skimage.io.imsave(file_out, texture)

if __name__ == '__main__':
    main()
