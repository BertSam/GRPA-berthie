#%%
from scipy.io import wavfile

import numpy as np
import librosa
import math
import scipy
import matplotlib.pyplot as plt

from spectrum import poly2lsf, lsf2poly
from statistics import mean
from scipy.signal import lfilter


class vocoder:
    def __init__(self, input_frame, ordre):
        super().__init__()
        mean = 0
        var = 1/100 # (40db) (À confirmer)
        std = math.sqrt(var)
        mu, sigma = mean, std # mean and standard deviation
        #frame_size = input_frame.size()
        self.frame_size = len(input_frame)
        self.wgn = np.random.normal(mu, sigma, self.frame_size)
        self.order = ordre
        self.input_frame = input_frame

    def get_lpc(self):

        frame = self.input_frame + self.wgn

        self.lpc_frame = librosa.core.lpc(frame, self.order)

        return self.lpc_frame

    def get_lsf(self):
        lpc = self.get_lpc()
        self.lsf = poly2lsf(lpc)

        return self.lsf

    # Calcul du résidu de prédiction
    def get_resRMS_lvl(self):
        residu = np.zeros(self.frame_size)
        residu = lfilter(self.get_lpc(),1,self.input_frame)

        self.resRMS = np.sqrt(np.mean(residu**2))
    
        return self.resRMS
    
        
fs, audioIn = wavfile.read('./NO_S.wav')

ts = 1/fs

temp_des_seq = 10/1000  # durée désirée de la séquence

audioIn = audioIn[0:int(temp_des_seq*fs)].astype(np.float) # Garder juste une seconde, pour aller plus vite + cast vers float

max_sig = max(abs(audioIn))

mean_sig = mean(audioIn)


audioIn = audioIn - mean_sig
audioIn = audioIn/max_sig

t = np.arange(0, len(audioIn))/fs

# plt.plot(t, audioIn, label='Signal source')
# plt.legend()
# plt.xlabel('Temps (s)')
# plt.axis([0,0.01,-1,1])
# plt.show()

# Analyse lpc

ordre_lpc = 5

vocoder = vocoder(audioIn, ordre_lpc)

resRMS_lvl = vocoder.get_resRMS_lvl()

print(resRMS_lvl)
# plt.plot(t, audioIn, label='Signal source')
# plt.plot(t, pred, label='Signal pred')
# plt.legend()
# plt.xlabel('Temps (s)')
# plt.axis([0,0.01,-1,1])
# plt.show()







