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
    input_file = h5py.File(file_name)
    dataset = input_file["postprocessing/visibility_map"][
        0, min_pixel:max_pixel, ...]
    if len(dataset.shape) > 1:
        dataset = np.mean(dataset, axis=0)
    average = np.mean(dataset)
    print(average)
    input_file.close()
    plt.plot(dataset)
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
