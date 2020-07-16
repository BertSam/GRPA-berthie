import utils

import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted

from os import listdir
from os.path import join

import librosa
import math
import scipy
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from spectrum import poly2lsf, lsf2poly
import numpy as np
import torchaudio

                          

import pyworld as pw

class FolderDataset(Dataset):

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        file_names = natsorted(
            [join(path, file_name) for file_name in listdir(path)]
        )
        self.file_names = file_names[
            int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.file_names[index], sr=None, mono=True)
        # print(np.mean(seq))
        # print(torch.mean(utils.linear_quantize(torch.from_numpy(seq), self.q_levels).float()))
        # plt.plot(seq)
        # plt.show()
        # plt.plot(torch.cat([
        #     torch.LongTensor(self.overlap_len) \
        #          .fill_(utils.q_zero(self.q_levels)),
        #     utils.linear_quantize(
        #         torch.from_numpy(seq), self.q_levels
        #     )
        # ]))
        # plt.show()
        # exit()
        return torch.cat([
            torch.LongTensor(self.overlap_len) \
                 .fill_(utils.q_zero(self.q_levels)),
            utils.linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

    def __len__(self):
        return len(self.file_names)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, overlap_len, M,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.M = M

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()

            # Fenêtres pour le future traitement des lpc
            han_win = np.hanning(self.overlap_len*3)
            han_win_edge = np.hanning(self.overlap_len*2)

            pitch = []
            pitch = torch.tensor(pitch, dtype=torch.float64)
            lsf = torch.empty((batch_size, int(n_samples / self.overlap_len), self.M), dtype=torch.float64)
            reslvlRMS = torch.empty((batch_size, int(n_samples / self.overlap_len)), dtype=torch.float64)
            voicing = torch.empty((batch_size, int(n_samples / self.overlap_len)), dtype=torch.float64)

            for batch_ind in range(batch_size):
                #temp_seq = utils.linear_dequantize(batch[batch_ind,:], 256).double().numpy()
                temp_seq = batch[batch_ind,:].double().numpy()

                # extraction du pitch
                fs = 16000
                _f0, t = pw.dio(temp_seq, fs)    # raw pitch extractor
                f0 = pw.stonemask(temp_seq, _f0, t, fs)  # pitch refinement
                ind = t * fs
                pitch = torch.cat([pitch, torch.tensor(f0[1::2], dtype=torch.float64)])

                # Extraction des lpc pour une trame "wind" (sur wind - overlap_len  à  wind + 2*overlap_len)
                res = []
                res = torch.tensor(res, dtype=torch.float64)
                lsf_frame = []
                lsf_frame = torch.tensor(lsf_frame, dtype=torch.float64)
                voicing_flag = []
                voicing_flag = torch.tensor(voicing_flag, dtype=torch.float64)


                for seq_begin in range(0, n_samples, self.overlap_len):

                    if seq_begin == 0:
                        from_index = seq_begin
                        to_index = seq_begin + 2 * self.overlap_len
                        trame = temp_seq[from_index : to_index] * han_win_edge # Hanning Windowing

                    elif (seq_begin + self.overlap_len) == n_samples:
                        from_index = seq_begin - self.overlap_len
                        to_index = seq_begin + self.overlap_len
                        trame = temp_seq[from_index : to_index] * han_win_edge # Hanning Windowing
                    else:
                        from_index = seq_begin - self.overlap_len
                        to_index = seq_begin + self.overlap_len * 2
                        trame = temp_seq[from_index : to_index] * han_win # Hanning Windowing

                    # Calcul du bruit blanc gaussien à ajouter au signal
                    rms = np.sqrt(np.mean(temp_seq**2))
                    var = rms * 0.0001 # (-40db)
                    std = math.sqrt(var)
                    mu, sigma = 0, std # mean = 0 and standard deviation
                    wgn = np.random.normal(mu, sigma, len(trame)) # génération bruit blanc gaussien

                    trame = trame + wgn # ajout de bruit blanc gaussien pour éviter "ill-conditioning"

                    # Conversion vers tenseur de lsf
                    a = librosa.core.lpc(trame , self.M)
                    lsf_frame = torch.cat([lsf_frame, torch.from_numpy(np.asarray(poly2lsf(a)))])

                    # Calcul de l'énergie résiduelle et conversion en tenseur
                    residu = lfilter(a, 1, trame)
                    res = torch.cat([res, torch.tensor([np.sqrt(np.mean(residu**2))], dtype=torch.float64)])

                    # Détection du voisement
                    voicing_frame_anal = temp_seq[seq_begin  : seq_begin + self.overlap_len]
                    voicing_frame_anal = voicing_frame_anal - np.mean(voicing_frame_anal)
                    zero_crossings_counter = len(np.where(np.diff(np.sign(voicing_frame_anal)))[0])

                    if zero_crossings_counter <= 20 and zero_crossings_counter >= 3:
                        flag_voice = torch.tensor([1], dtype=torch.float64)
                    else:
                        flag_voice = torch.tensor([0], dtype=torch.float64)

                    voicing_flag = torch.cat([voicing_flag, flag_voice])

                lsf_frame = torch.reshape(lsf_frame, [-1, self.M])

                lsf[batch_ind,:,:] = lsf_frame
                reslvlRMS[batch_ind, :] = res
                voicing[batch_ind, :] = voicing_flag

            pitch = torch.reshape(pitch, [batch_size, -1])

            
            # correction pour cohérence pitch et voisement
            [_, size_] = pitch.size()
            for batch_ind in range(batch_size):
                for i in range(size_):
                    if voicing[batch_ind, i] == 0 or pitch[batch_ind, i] == 0:
                        pitch[batch_ind, i] = 50 # valeur de pitch minimal (arbitraire)
                    

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):

                #Extraction des param de la trame correspondante dans la séquence
                trame_ind = torch.arange(int(seq_begin/self.overlap_len), int((seq_begin+self.seq_len)/self.overlap_len)).int().numpy()
                input_lsfs = torch.reshape(lsf[:, trame_ind, :], (batch_size, -1))
                input_reslvls = reslvlRMS[:, trame_ind]
                input_pitchs = pitch[:, trame_ind]
                input_voicings = voicing[:, trame_ind]

                # Quantification des params
                # Q lsf
                input_lsfs = input_lsfs*20000 # max input_lsfs ish = 3.2 (overshoot), donc *20000 = max à 64 000 sur 65 536 (bonne résoultion)
                input_lsfs = input_lsfs.long()

                # Q reslvls
                input_reslvls = input_reslvls*650000 # max reslvls ish = 0.1 (overshoot), donc *650000 = max à 65 000 sur 65 536 (bonne résoultion)
                input_reslvls = input_reslvls.long()
          
                # Q pitch 
                input_pitchs = input_pitchs*100 # max pitch = 500, donc *100 = max à 50 000 sur 65 536 (bonne résoultion)
                input_pitchs = input_pitchs.long()

                # Q voicings
                input_voicings = input_voicings.long()


                # Création des vecteur de conditionnement (pour toute la sequence)
                hf_long = torch.cat([input_lsfs, input_reslvls, input_pitchs, input_voicings],1)


                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len

                sequences = batch[:, from_index : to_index]

                input_sequences = sequences[:, : -1]

                target_sequences = sequences[:, self.overlap_len :].contiguous()

                input_sequences = torch.cat([input_sequences, hf_long],1)
        
                yield (input_sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()
