import numpy as np
import skimage.io


def synthesize_texture(in_image,
                       out_shape=(512, 512),
                       N=1, M=20, polish=3,
                       random_seed=1):

    # Initialize numpy's random:
    np.random.seed(random_seed)

    # Create an extra dimension if the input image is in grayscale:
    if len(in_image.shape) < 3:
        in_image = in_image[:, :, np.newaxis]

    # Prepare the output image and place the initial seed pixels:
    in_height, in_width, in_channels = in_image.shape
    in_size = in_height * in_width

    out_height, out_width = out_shape
    out_size = out_width * out_height

    sample = in_image.reshape((in_width * in_height, in_channels))

    def metric(a, b, sigma=20.0):
        sigma_inv = 1.0 / sigma
        r = 1.0 + sigma_inv * (a - b) ** 2
        return -np.log(np.prod(r))

    origins = -1 * np.ones(out_size)
    shuffle = np.random.permutation(out_size)

    for iteration in range(polish + 1):
        print 'Iteration no. {}'.format(iteration)

        for counter in range(out_size):
            if counter % (out_size // 100) == 0:
                print 'Percent: {}'.format(100 * counter / out_size)

            f = shuffle[counter]
            fx, fy = f % out_width, f // out_width

            neighborsNumber = 8 if iteration > 0 else min(8, counter)
            neighborsFound = 0

            candidates = np.empty(neighborsNumber + M, dtype=int)

            if neighborsNumber > 0:

                neighbors = np.empty(neighborsNumber, dtype=int)
                x, y = np.empty(4, dtype=int), np.empty(4, dtype=int)
                radius = 1
                while neighborsFound < neighborsNumber:
                    x[0] = fx - radius
                    y[0] = fy - radius
                    x[1] = fx - radius
                    y[1] = fy + radius
                    x[2] = fx + radius
                    y[2] = fy + radius
                    x[3] = fx + radius
                    y[3] = fy - radius

                    for k in range(2 * radius):
                        for d in range(4):
                            x[d] = (x[d] + 10 * out_width) % out_width
                            y[d] = (y[d] + 10 * out_height) % out_height

                            if neighborsFound >= neighborsNumber:
                                continue

                            point = x[d] + y[d] * out_width
                            if origins[point] != -1:
                                neighbors[neighborsFound] = point
                                neighborsFound += 1

                        y[0] += 1
                        x[1] += 1
                        y[2] -= 1
                        x[3] -= 1

                    radius += 1

                for n in range(neighborsNumber):
                    cx = (origins[neighbors[n]] + (f - neighbors[n]) % out_width + 100 * in_width) % in_width
                    cy = (origins[neighbors[n]] // in_width + f // out_width - neighbors[n] // out_width + 100 * in_height) % in_height
                    candidates[n] = cx + cy * in_width

            for m in range(M):
                candidates[neighborsNumber + m] = np.random.randint(in_size)

            maximum = -1e+10
            argmax = -1

            for c in range(len(candidates)):
                sum = 0
                ix, iy = candidates[c] % in_width, candidates[c] // in_width

                for dy in range(-N, N+1):
                    for dx in range(-N, N+1):
                        if dx != 0 or dy != 0:
                            SX = (ix + dx) % in_width
                            SY = (iy + dy) % in_height

                            FX = (fx + dx) % out_width
                            FY = (fy + dy) % out_height

                            S = SX + SY * in_width
                            F = FX + FY * out_width

                            origin = origins[F]
                            if origin != -1:
                                sum += metric(sample[origin, :],
                                              sample[S, :])

                if sum >= maximum:
                    maximum = sum
                    argmax = candidates[c]

            origins[f] = argmax

    # Save
    out_image = np.zeros((out_height, out_width, in_channels))

    for index in range(out_size):
        row, col = index % out_width, index // out_width
        out_image[row, col, :] = sample[origins[index]]

    return out_image


def main():
    file_in = '../../imgs/textures/samples/classical1.png'
    file_out = '../../imgs/textures/synthesized/resynthesizer.png'

    out_width = 256
    out_height = 256
    half_window = 1
    random_points = 20
    polish = 0
    random_seed = 0

    input_patch = skimage.io.imread(file_in)
    input_patch = skimage.img_as_float(input_patch)
    texture = synthesize_texture(input_patch,
                                 (out_height, out_width),
                                 half_window,
                                 random_points,
                                 polish,
                                 random_seed)
    if texture.shape[2] == 1:
        texture = texture.squeeze()

    skimage.io.imsave(file_out, texture)


if __name__ == '__main__':
    main()
