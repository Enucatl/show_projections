import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py
from scipy import stats
from scipy.signal import argrelextrema

parser = argparse.ArgumentParser(
    __doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--min_pixel",
    type=int,
    nargs="?", default=0,
    help="minimum pixel")

parser.add_argument(
    "--max_pixel",
    type=int,
    nargs="?", default=1024,
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
    group_name = "raw_images"
    input_file = h5py.File(file_name)
    datasets = [dataset[min_pixel:max_pixel, ...]
                for dataset in input_file[group_name].values()
                if isinstance(dataset, h5py.Dataset)]
    stacked = np.concatenate(datasets, axis=0)
    image = stacked
    limits = stats.mstats.mquantiles(
        image,
        prob=[0.02, 0.98])
    print("black={0[0]}, white={0[1]}".format(limits))
    input_file.close()
    plt.figure()
    summed = np.mean(stacked, axis=1)
    plt.plot(summed)
    plt.figure()
    plt.imshow(image, interpolation="none", aspect='auto')
    plt.clim(*limits)
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
