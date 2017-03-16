import click
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import stats

@click.command()
@click.argument("input_file_name", type=click.Path(exists=True))
def main(input_file_name):
    input_file = h5py.File(input_file_name)
    dataset = input_file["postprocessing/visibility"][...]
    print(dataset.shape)
    dataset = np.median(dataset, axis=-1)
    dataset = np.median(dataset, axis=-1)
    input_file.close()
    print(dataset.shape)
    plt.plot(dataset.T)
    plt.ion()
    plt.show()
    input("Press ENTER to quit.")

if __name__ == "__main__":
    main()
