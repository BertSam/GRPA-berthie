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
import scipy.fftpack as sf


class vocoder:
    def __init__(self, input_frame, samp_rate):
        super().__init__()
        mean = 0
        var = 1/100 # (40db) (À confirmer)
        std = math.sqrt(var)
        mu, sigma = mean, std # mean and standard deviation
        #frame_size = input_frame.size()
        self.sampling_rate = samp_rate
        self.frame_size = len(input_frame)
        self.wgn = np.random.normal(mu, sigma, self.frame_size)
        
        self.input_frame = input_frame

    def get_lpc(self, ordre):
        self.order = ordre
        frame = self.input_frame + self.wgn

        self.lpc_frame = librosa.core.lpc(frame, self.order)

        return self.lpc_frame

    def get_lsf(self, ordre):
        lpc = self.get_lpc(ordre)
        self.lsf = poly2lsf(lpc)

        return self.lsf

    # Calcul du résidu de prédiction
    def get_resRMS_lvl(self):
        residu = np.zeros(self.frame_size)
        residu = lfilter(self.get_lpc(),1,self.input_frame)

        self.resRMS = np.sqrt(np.mean(residu**2))
    
        return self.resRMS

    # def get_pitch(self, Low_cutoff, High_cutoff):   

    #     N = self.input_frame.size

    #     wind = np.hanning(N) 

    #     spectrum = np.fft.rfft(wind*self.input_frame, n=N) 
        
    #     f = np.zeros(int(N/2))
    #     for i in range(0,int(N/2)):
    #         f[i] = i
         
    #     f = self.sampling_rate*f/N
    #     magn_spectrum = abs(spectrum/N)

    #     print(len(f))
    #     print(len(magn_spectrum[0:-1]))

    #     plt.plot(f,magn_spectrum[0:-1]) 


    #     plt.show()


    #     [Low_cutoff, High_cutoff, self.sampling_rate] = map(float, [Low_cutoff, High_cutoff, self.sampling_rate])

    #     #Convert cutoff frequencies into points on spectrum
    #     [Low_point, High_point] = map(lambda F: F/self.sampling_rate * self.input_frame.size, [Low_cutoff, High_cutoff])
        
    #     pitch = np.where(Spectrum == np.max(Spectrum[Low_point : High_point])) # Calculating which frequency has max power.

    #     return pitch


    
        
fs, audioIn = wavfile.read('./NO_S.wav')

ts = 1/fs

temp_des_seq = 10/1000  # durée désirée de la séquence

start_time = 0.5

stop_time = start_time + temp_des_seq

audioIn = audioIn[int(start_time*fs):int(stop_time*fs)].astype(np.float) # Garder juste une seconde, pour aller plus vite + cast vers float

max_sig = max(abs(audioIn))

mean_sig = mean(audioIn)


audioIn = audioIn - mean_sig
audioIn = audioIn/max_sig

t = np.arange(0, len(audioIn))/fs

plt.plot(t, audioIn, label='Signal source')
plt.legend()
plt.xlabel('Temps (s)')
plt.axis([0,0.01,-1,1])
plt.show()

# Analyse lpc

ordre_lpc = 5


vocoder = vocoder(audioIn, fs)

lpc = vocoder.get_lpc(ordre_lpc)
lsf = vocoder.get_lsf(ordre_lpc)

pitch = vocoder.get_pitch(float(60), float(250))
print(pitch)


# plt.plot(t, audioIn, label='Signal source')
# plt.plot(t, pred, label='Signal pred')
# plt.legend()
# plt.xlabel('Temps (s)')
# plt.axis([0,0.01,-1,1])
# plt.show()







