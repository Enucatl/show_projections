import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        __doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "file",
        nargs=1,
        help="hdf5 file with the dpc reconstruction")
    args = parser.parse_args()
    input_file = h5py.File(args.file[0], "r")
    deconvolved = input_file["postprocessing/deconvolved"]
    dataset = np.zeros(deconvolved.shape[:-1] + (3, ))
    _, images = plt.subplots(3, 1, sharex=True)
    padded = np.zeros(deconvolved.shape[:-1] +
                      (deconvolved.shape[-1] + 1,))
    padded[..., :-1] = deconvolved[...]
    padded[..., -1] = deconvolved[..., 0]
    dataset[..., 0] = np.sum(padded, axis=-1)
    angles = np.linspace(-np.pi, np.pi, padded.shape[-1])
    dataset[..., 1] = np.dot(padded, angles)
    dataset[..., 2] = stats.mstats.moment(
        padded,
        moment=2,
        axis=-1)
    for i, image in enumerate(images):
        data = dataset[..., i].T
        print(data)
        print(data.shape)
        limits = stats.mstats.mquantiles(
            data,
            prob=[0.02, 0.98])
        image.axis("off")
        matplotlib_image = image.imshow(data, interpolation="none")
        matplotlib_image.set_clim(*limits)
    # print(dataset)
    plt.tight_layout()
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")
