import torch
import numpy as np
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')

def save_logs(status_log):
    fname = os.path.join('../result/logs', 'ddpm_log.csv')
    f=open(fname,'ab')
    # header = ['Time', 'Epoch', 'Step', 'Loss']
    with open(fname, 'ab') as f:
        np.savetxt(f, np.array(status_log).reshape(1, -1), fmt="%s", delimiter=',', comments='')

def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
