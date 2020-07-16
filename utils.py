import torch
from torch import nn
import numpy as np

import librosa
import math
import scipy
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from spectrum import poly2lsf, lsf2poly

import pyworld as pw


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    
    # samples /= samples.abs().max()
    # samples -= samples.mean()
    # samples *= q_zero(q_levels)
    # samples += q_zero(q_levels)
    return samples.long()

def my_normalize(samples):
    return (samples - torch.mean(samples)) / torch.std(samples)


def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1
    
    # samples = samples.clone()

    # plt.figure(0)
    # plt.plot(samples.cpu())

    # samples -= q_zero(q_levels)
    # samples /= q_zero(q_levels)

    # plt.figure(1)
    # plt.plot(samples.cpu().float())

    # plt.show()

    # return samples.float() 

def q_zero(q_levels):
    return q_levels // 2

def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]
    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))

    ##############   
def get_vocoder_param(input_frames, order):
    [batch_size, frame_size] = input_frames.size()
    voc_param = torch.zeros(batch_size, order+1)
    wind = np.hanning(frame_size)
    

    for batch in range(batch_size): 

        temp_seq = input_frames[batch,:].float().cpu().numpy()

        # Calcul du bruit blanc gaussien à ajouter au signal 
        rms = np.sqrt(np.mean(temp_seq**2))
        var = rms * 0.0001 # (-40db)
        std = math.sqrt(var)
        mu, sigma = 0, std # mean = 0 and standard deviation
        wgn = np.random.normal(mu, sigma, frame_size)


        # hanning windowing
        temp_seq = temp_seq*wind

        a = librosa.core.lpc(temp_seq  + wgn, order)
        lsf_frame = torch.from_numpy(np.asarray(poly2lsf(a)))


        residu = lfilter(a, 1, temp_seq)

        reslvlRMS = torch.tensor([20*math.log10(np.sqrt(np.mean(residu**2)))], dtype=torch.float64)
        # print(torch.cat((lsf_frame, reslvlRMS)))
        voc_param[batch,:] = linear_quantize(torch.cat((lsf_frame, reslvlRMS)), 256)

    return voc_param


