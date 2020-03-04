import numpy as np
import matplotlib.pyplot as plt
import torch
import os

""" 
This is a version of: https://github.com/gpapamak/maf/blob/master/datasets/power.py, 
adapted to work with Python 3.x and PyTorch. 
"""

batch_size = 100


class POWER:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data_normalised()

        self.train = self.Data(trn)
        self.val = self.Data(val)
        self.test = self.Data(tst)

        self.n_dims = self.train.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError("Invalid data split")

        util.plot_hist_marginals(data_split.x)
        plt.show()


def load_data(root="data/maf_data/"):
    return np.load(root + "power/data.npy")


def load_data_split_with_noise():

    rng = np.random.RandomState(42)

    data = load_data()
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    ############################
    # Add noise
    ############################
    # global_intensity_noise = 0.1*rng.rand(N, 1)
    voltage_noise = 0.01 * rng.rand(N, 1)
    # grp_noise = 0.001*rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised():

    data_train, data_validate, data_test = load_data_split_with_noise()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


data = POWER()
train = torch.from_numpy(data.train.x)
val = torch.from_numpy(data.val.x)
test = torch.from_numpy(data.test.x)
n_in = data.n_dims

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

print(len(train_loader.dataset))

