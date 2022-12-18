'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
print(input.size(), target.size())
print(input)
print(target)
output = loss(input, target)
'''

'''
from pynvml import *
nvmlInit()
deviceCount = nvmlDeviceGetCount()
print(deviceCount)

for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("Device", i, ":", nvmlDeviceGetName(handle))
'''

import pynvml
pynvml.nvmlInit()
# 这里的1是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total) #第二块显卡总的显存大小
print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print(meminfo.free) #
