import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np
import librosa
import math
import scipy
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from spectrum import poly2lsf, lsf2poly


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        # print('============================')
        # print(frame_sizes)
        # print('============================')
        # print(np.cumprod(frame_sizes))
        # print('============================')

        ns_frame_samples = map(int, np.cumprod(frame_sizes))

        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])

        # print('============================')
        # print(frame_sizes[0])
        # print('============================')

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

        # print('============================')
        # print(list(self.frame_level_rnns))
        # print('============================')

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, weight_norm):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim

        # Longueur de vecteur de conditionnement 
        #self.M = M 

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.input_expand.weight)
        init.constant(self.input_expand.bias, 0)
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)

        #Tentative d'inclure le conditioning BGF (20-06-08)
        self.input_conditioning = torch.nn.Conv1d(
           in_channels = 24,
           out_channels = dim,
           kernel_size = 1
        )  

        init.kaiming_uniform(self.input_conditioning.weight)
        init.constant(self.input_conditioning.bias, 0)  
        if weight_norm:
           self.input_conditioning = torch.nn.utils.weight_norm(self.input_conditioning)
        

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal]
            )
            init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )
        init.uniform(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(
                self.upsampling.conv_t
            )
    #def forward(self, prev_samples, upper_tier_conditioning, hidden):
    def forward(self, prev_samples, hf, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()


        input_frame = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)

        input_hf = self.input_conditioning(
          hf.permute(0, 2, 1)
        ).permute(0, 2, 1)

      
        input = input_frame + input_hf


        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        (output, hidden) = self.rnn(input, hidden)

        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return (output, hidden)


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        init.kaiming_uniform(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        #Tentative d'inclure le conditioning BGF (20-06-08)
        # self.input_conditioning = torch.nn.Conv1d(
        #    in_channels = 24,
        #    out_channels = dim,
        #    kernel_size = 1
        # )  
        # init.kaiming_uniform(self.input_conditioning.weight)
        # init.constant(self.input_conditioning.bias, 0)  
        # if weight_norm:
        #    self.input_conditioning = torch.nn.utils.weight_norm(self.input_conditioning)
        #

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.hidden.weight)
        init.constant(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)
    #def forward(self, prev_samples, hf, upper_tier_conditioning):
    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)
        #hf = hf.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()
        ## Ajout dim=1 BGF (20-06-08)
        temp = x.view(-1, self.q_levels)
        return F.log_softmax(temp) \
                .view(batch_size, -1, self.q_levels)
        ##

class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    #def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
    def run_rnn(self, rnn, prev_samples, hf, upper_tier_conditioning):    
        #print("IN: run_rnn()")
        (output, new_hidden) = rnn(
            prev_samples, hf, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        (batch_size, _) = input_sequences.size()

        tier_ratio = np.zeros(len(self.model.frame_level_rnns))
        upper_tier_conditioning = None
        for c, rnn in enumerate(reversed(self.model.frame_level_rnns)):

            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )
            
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )

            [batch_size_temp, layer_temp, _] = prev_samples.size()
            
            tier_ratio[c] = (self.model.frame_level_rnns[len(self.model.frame_level_rnns)-c-1].n_frame_samples) / (self.model.frame_level_rnns[len(self.model.frame_level_rnns)-c-2].n_frame_samples)
             

            if upper_tier_conditioning is None:
                hf = torch.zeros([batch_size_temp, layer_temp, 24], dtype=torch.float)
                for batch in range(batch_size_temp):
                    for layer  in range(layer_temp):
                        temp_input = prev_samples[batch,layer,:]
                        
                        temp_lpc = vocoder(temp_input).get_lsf()
                        hf[batch, layer, :] = torch.tensor(temp_lpc)
                        top_hf = hf
 
            elif  upper_tier_conditioning is not None:
                upper_tier_hf = hf
                hf = torch.zeros([batch_size_temp, layer_temp, 24], dtype=torch.float)
                for batch in range(batch_size_temp):
                    for layer  in range(layer_temp): 
                        hf[batch, layer, :] = upper_tier_hf[batch, math.floor(layer/(tier_ratio[c-1])), :]

            hf = hf.cuda()
            
            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, hf, upper_tier_conditioning
            )
            
        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences \
            [:, self.model.lookback - bottom_frame_size :]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len):
        # generation doesn't work with CUDNN for some reason
        torch.backends.cudnn.enabled = False

        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len) \
                         .fill_(utils.q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]


        tier_ratio = np.zeros(len(self.model.frame_level_rnns))
        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                prev_samples = torch.autograd.Variable(
                    2 * utils.linear_dequantize(
                        sequences[:, i - rnn.n_frame_samples : i],
                        self.model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                #print(rnn)
                #print(tier_index)
               # print(prev_samples.size())


                if self.cuda:
                    prev_samples = prev_samples.cuda()
                

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)

                ###########

                #print(prev_samples.size())
                [batch_size_temp, layer_temp, _] = prev_samples.size()
               
                tier_ratio[tier_index] = (self.model.frame_level_rnns[len(self.model.frame_level_rnns)-tier_index-1].n_frame_samples) / (self.model.frame_level_rnns[len(self.model.frame_level_rnns)-tier_index-2].n_frame_samples)
            

                if upper_tier_conditioning is None:
                    hf = torch.zeros([batch_size_temp, layer_temp, 24], dtype=torch.float)
                    for batch in range(batch_size_temp):
                        for layer  in range(layer_temp):
                            temp_input = prev_samples[batch,layer,:]
                            
                            temp_lpc = vocoder(temp_input).get_lsf()
                            hf[batch, layer, :] = torch.tensor(temp_lpc)
    
                elif  upper_tier_conditioning is not None:
                    upper_tier_hf = hf
                    hf = torch.zeros([batch_size_temp, layer_temp, 24], dtype=torch.float)
                    for batch in range(batch_size_temp):
                        for layer  in range(layer_temp): 
                            # print('##########')
                            # print(batch)
                            # print('---')
                            # print(layer)
                            #print(math.floor(layer/(tier_ratio[tier_index-1])))
                            if layer==0:
                                hf[batch, layer, :] = upper_tier_hf[batch, 0, :]
                            else:    
                                hf[batch, layer, :] = upper_tier_hf[batch, math.floor(layer/(tier_ratio[tier_index-1])), :]

                hf = hf.cuda()

                #####


                #print(prev_samples.size())
                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, hf, upper_tier_conditioning
                )

            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size : i],
                volatile=True
            )
            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                                      .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]


class vocoder:
    def __init__(self, input_frame):
        super().__init__()
        mean = 0
        var = 1/100 # (40db) (À confirmer)
        std = math.sqrt(var)
        mu, sigma = mean, std # mean and standard deviation
        self.input_frame = input_frame
        self.frame_size = self.input_frame.size()
        self.wgn = np.random.normal(mu, sigma, self.frame_size)
        self.order = 24
        
    def get_lpc(self):
        self.input_frame = self.input_frame.cpu().numpy()
        self.input_frame = self.input_frame + self.wgn

        self.lpc_frame = librosa.core.lpc(self.input_frame, self.order)

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




