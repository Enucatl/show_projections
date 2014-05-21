import numpy as np
import h5py
from functools import reduce
import operator
from progress_bar.progress_bar import progress_bar as bar
from pymatbridge import Matlab
mlab = Matlab()
mlab.start()


def deconvlucy(i, psf, numit=10, dampar=0,
               weight=None, readout=0, subsmpl=1):
    if weight is None:
        weight = np.ones_like(i)
    """Calls the deconvlucy matlab function
    see http://www.mathworks.ch/ch/help/images/ref/deconvlucy.html

    """
    results = mlab.run_code(
        "deconvlucy({0}, {1}, {2}, {3}, {4}, {5}, {6})".format(
            i.tolist(), psf.tolist(), numit,
            dampar, weight.tolist(), readout, subsmpl))
    output_string = [line
                     for line in results["content"]["stdout"].splitlines()
                     if line
                     and "Columns" not in line
                     and "ans" not in line]
    clean_string = " ".join("".join(output_string).split())
    output_array = np.fromstring(clean_string, sep=" ")
    return output_array

try:
    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(
            __doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "file",
            nargs=1,
            help="hdf5 file with the dpc reconstruction")
        parser.add_argument(
            "--numit",
            nargs="?",
            default=10,
            help="iterations for the Lucy-Richardson algorithm")
        args = parser.parse_args()
        input_file = h5py.File(args.file[0])
        numit = args.numit
        sample_curves = input_file[
            "postprocessing/phase_stepping_curves"][0, 300:850, ...]
        flat_curves = input_file[
            "postprocessing/flat_phase_stepping_curves"][0, 300:850, ...]
        n_curves = reduce(operator.mul, sample_curves.shape[:-1])
        n_steps = sample_curves.shape[-1]
        deconvolved_reconstruction = np.zeros((n_curves, n_steps))
        for i, (sample, flat) in enumerate(zip(
                np.reshape(sample_curves, (n_curves, n_steps)),
                np.reshape(flat_curves, (n_curves, n_steps)))):
            print(bar((i + 1) / n_curves), end="\r")
            #print(sample, flat)
            #if i > 1:
                #break
            deconvolved_reconstruction[i] = deconvlucy(
                sample, flat, numit=numit)
        deconvolved_reconstruction = np.reshape(
            deconvolved_reconstruction, sample_curves.shape)
        #print(deconvolved_reconstruction.shape)
        if "postprocessing/deconvolved" in input_file:
            del input_file["postprocessing/deconvolved"]
        input_file["postprocessing/deconvolved"] = deconvolved_reconstruction
        input_file.close()
finally:
    mlab.stop()
