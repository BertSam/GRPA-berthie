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

    def get_pitch(self, Low_cutoff, High_cutoff):  
        pitches, magnitudes = librosa.core.piptrack(y=self.input_frame, window='hann', sr=self.sampling_rate, fmin=Low_cutoff, fmax=High_cutoff)
        index = magnitudes.argmax()
        self.pitch = pitches[index]

        return self.pitch


    
        
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

f0 = 125
w0 = 2*math.pi*f0
amplitude   = np.sin(w0*t)

#pitches, magnitudes = librosa.piptrack(y=audioIn.astype(np.float), sr=fs, fmin=60, fmax=300)
pitches, magnitudes = librosa.piptrack(y=audioIn, sr=fs, fmin=50, fmax=300, n_fft=1024)



print('*****************************')
print(pitches[np.nonzero(pitches)])
print('*****************************')
print(magnitudes[np.nonzero(pitches)])

plt.plot(t, amplitude)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


# Analyse lpc

ordre_lpc = 5


# vocoder = vocoder(audioIn, fs)

# lpc = vocoder.get_lpc(ordre_lpc)
# lsf = vocoder.get_lsf(ordre_lpc)
# pitch = vocoder.get_pitch(60, 600)

# print(pitch)



#et x values of the sine wave
 
# Amplitude ofthe sine wave is sine of a variable like time

# f0 = 1000
# w0 = 2*math.pi*f0
# amplitude   = np.sin(w0*t)

# # Plot a sine wave using time and amplitude obtained for the sine wave
# plt.plot(t, amplitude)
# # Give a title for the sine wave plot
# plt.title('Sine wave')
# # Give x axis label for the sine wave plot
# plt.xlabel('Time')
# # Give y axis label for the sine wave plot
# plt.ylabel('Amplitude = sin(time)')
# plt.grid(True, which='both')
# plt.axhline(y=0, color='k')
# plt.show()


# vocoder = vocoder(amplitude, fs)

# pitch = vocoder.get_pitch(60, 2000)

# print(pitch)
