import os

from tqdm.contrib.concurrent import process_map
import h5py

if __name__ == '__main__':
    f = h5py.File(os.path.join("data", "raw", "opmotion", "opmotion.h5"), "r")
    color = f["colors"][()]
    mean_color = color.reshape(-1, 3).mean(axis=0)
    std_coloar = color.reshape(-1, 3).std(axis=0)
    # mean_color = mean_color.mean(axis=0)

    print(mean_color)