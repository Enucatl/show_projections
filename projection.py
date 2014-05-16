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
    "file",
    nargs=1,
    help="hdf5 file with the raw data")

if __name__ == '__main__':
    args = parser.parse_args()
    file_name = args.file[0]
    min_pixel = args.min_pixel
    max_pixel = args.max_pixel
    group_name = "/"
    input_file = h5py.File(file_name)
    datasets = [dataset
                for dataset in input_file[group_name].values()
                if isinstance(dataset, h5py.Dataset)]
    stacked = np.dstack(datasets)
    image = stacked[0, min_pixel:max_pixel].T
    limits = stats.mstats.mquantiles(
        image,
        prob=[0.02, 0.98])
    plt.imshow(image, interpolation="none")
    plt.clim(*limits)
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
