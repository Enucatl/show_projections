import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py
from scipy import stats

parser = argparse.ArgumentParser(
    __doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--min_pixel",
    nargs="?", default=0,
    type=int,
    help="minimum pixel")

parser.add_argument(
    "--max_pixel",
    type=int,
    nargs="?", default=1280,
    help="maximum pixel")

parser.add_argument(
    "file",
    nargs='+',
    help="hdf5 file(s) with the raw data")

if __name__ == '__main__':
    args = parser.parse_args()
    min_pixel = args.min_pixel
    max_pixel = args.max_pixel
    datasets = []
    for file_name in args.file:
        input_file = h5py.File(file_name)
        dataset = input_file["postprocessing/visibility_map"][
            0, min_pixel:max_pixel, ...]
        datasets.append(dataset)
    dataset = np.hstack(datasets)
    if dataset.shape[1] > 1:
        dataset = np.mean(dataset, axis=0)
    average = np.mean(dataset)
    print(average)
    input_file.close()
    plt.plot(dataset)
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
