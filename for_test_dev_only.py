import numpy as np
import librosa
import math
import scipy
import matplotlib.pyplot as plt
import torch



x = torch.tensor([0,1,2,3,4,5,6,7,8,9])

sz = list(x.size())
print(sz[0])
print(x[-1])
emd = sz[0] - 1
print(emd)
print(x[emd])

