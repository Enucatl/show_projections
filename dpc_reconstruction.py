import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py
from scipy import stats

parser = argparse.ArgumentParser(
    __doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "min_pixel",
    nargs="?", default=300,
    help="minimum pixel")

parser.add_argument(
    "max_pixel",
    nargs="?", default=850,
    help="maximum pixel")

parser.add_argument(
    "--dataset",
    nargs="?",
    default="postprocessing/dpc_reconstruction",
    help="dataset inside the hdf5 file")

parser.add_argument(
    "file",
    nargs=1,
    help="hdf5 file with the data")

if __name__ == '__main__':
    args = parser.parse_args()
    file_name = args.file[0]
    min_pixel = args.min_pixel
    max_pixel = args.max_pixel
    input_file = h5py.File(file_name)
    dataset = input_file[args.dataset][0, min_pixel:max_pixel, ...]
    input_file.close()
    _, images = plt.subplots(3, 1, sharex=True)
    correction = np.tile(
        np.mean(dataset[..., 1], axis=0),
        (dataset.shape[0], 1))
    dataset[..., 1] = dataset[..., 1] - correction
    for i, image in enumerate(images):
        data = dataset[..., i].T
        limits = stats.mstats.mquantiles(
            data,
            prob=[0.02, 0.98])
        image.axis("off")
        matplotlib_image = image.imshow(data, interpolation="none")
        matplotlib_image.set_clim(*limits)
    plt.tight_layout()
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
