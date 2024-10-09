import cupy as cp
import matplotlib.pyplot as plt
import mrcfile
import argparse

def average_along_z(tg_path, start=0, end=None):
    with mrcfile.open(tg_path) as mrc:
        data = mrc.data
    data = cp.array(data)
    if end is not None:
        data = data[start:end, :, :]
    # Average all the 2D images along z-axis
    average = cp.mean(data, axis=0)
    return average

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tomo_path', '-t', type=str)
    parser.add_argument('--start', '-s', type=int, default=0)
    parser.add_argument('--end', '-e', type=int, default=None)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    tg_path = args.tomo_path
    start = args.start
    end = args.end
    output = args.output

    average = average_along_z(tg_path, start, end)

    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(cp.asnumpy(average))

    # average = low_pass_filter(average, 20, pixel_size)
    # average = gaussian_filter(average, sigma=1)
    # Show the average image
    # plt.subplot(121)
    # plt.imshow(average.get(), cmap='gray')
    # plt.axis('off')
    # plt.subplot(122)
    # plt.imshow(average_part.get(), cmap='gray')
    # plt.axis('off')
    # plt.show()