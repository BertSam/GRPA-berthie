import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np
import librosa
import math
import scipy
import matplotlib.pyplot as plt


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels

        ns_frame_samples = map(int, np.cumprod(frame_sizes))

        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
            
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )


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

        # Tentative d'inclure le conditioning BGF (20-06-08)
        self.input_hf_vocoder = torch.nn.Conv1d(
            in_channels=24,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.input_hf_vocoder.weight)
        init.constant(self.input_hf_vocoder.bias, 0)
        if weight_norm:
            self.input_hf_vocoder = torch.nn.utils.weight_norm(self.input_hf_vocoder)
        #

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
        
    def forward(self, prev_samples, vocoder_conditioning, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()

        # tensor hf bidon représentant le données du conditionneur
        #hf_bidon = torch.zeros([batch_size,ToUnderstandVar,24]).cuda()


        print('lllllllllllll')
        print(prev_samples.size())
        print(type(prev_samples))
        print('--------------------------')
        print(vocoder_conditioning.size())
        print(type(vocoder_conditioning))
        print('lllllllllllll')

        
        ## Tentative d'inclure le conditioning BGF (20-06-10)
        input1_test = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)


        input2_test = self.input_hf_vocoder(
          vocoder_conditioning.permute(0, 2, 1)
        ).permute(0, 2, 1)

        print('lllllllllllll')
        print(prev_samples.size())
        print(type(prev_samples))
        print('--------------------------')
        print(vocoder_conditioning.size())
        print(type(vocoder_conditioning))
        print('lllllllllllll')

        input = input1_test + input2_test

        # Original
        # input = self.input_expand(
        #   prev_samples.permute(0, 2, 1)
        # ).permute(0, 2, 1)
        ##

        if upper_tier_conditioning is not None:
            input = input + upper_tier_conditioning
            print(upper_tier_conditioning.size())
            exit()

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

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

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

    def run_rnn(self, rnn, prev_samples, vocoder_conditioning, upper_tier_conditioning):
        (output, new_hidden) = rnn(
            prev_samples, vocoder_conditioning, upper_tier_conditioning, self.hidden_states[rnn]
        )
        
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)
        mean = 0
        var = 100
        std = math.sqrt(var)
        mu, sigma = mean, std # mean and standard deviation
        self.wgn = np.random.normal(mu, sigma, 160)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        (batch_size, _) = input_sequences.size()
        
        upper_tier_conditioning = None

        #modif start here (bgf 20-06-16)

        rnn_upper_tier = self.model.frame_level_rnns[-1]

        # print(input_sequences.size())

        # print(rnn_upper_tier.n_frame_samples)
       
        from_index = self.model.lookback - rnn_upper_tier.n_frame_samples
        to_index = -rnn_upper_tier.n_frame_samples + 1

        prev_samples = 2 * utils.linear_dequantize(
            input_sequences[:, from_index : to_index],
            self.model.q_levels
        )
        print(prev_samples.size())
        prev_samples = prev_samples.contiguous().view(
            batch_size, -1, rnn_upper_tier.n_frame_samples
            )
        print(prev_samples.size())
        exit()
        order = 23 

        #vocoder_conditioning = torch.zeros([batch_size, N_LAYER, order+1]).cuda()
    
        (_, N_LAYER, nb_sample) = prev_samples.size()
        for batch in range(batch_size):
            for layer in range(N_LAYER):
                temp = prev_samples[batch, layer, :].cpu().numpy()
                print(len(temp))
                print(len(self.wgn))
                temp = temp + self.wgn
                    
                lpc_coef_test = librosa.core.lpc(temp, order)
    
                # vocoder_conditioning[batch, layer , :] = torch.tensor(lpc_coef_test, dtype=torch.float).cuda()
    
            
        for rnn in reversed(self.model.frame_level_rnns):

            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1

            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )

            print(prev_samples.size())
        
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )
            print(prev_samples.size())       
            exit()
            # vocoder_conditioning = vocoder_conditioning.contiguous().view(
            #     batch_size, -1, 24
            # )

            
            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, vocoder_conditioning, upper_tier_conditioning
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
                print('pppppppppppp')
                print(prev_samples.size())
                print(type(prev_samples))
                print('pppppppppppp')
                [SIZE_1, SIZE_2, _] = prev_samples.size()
                vocoder_conditioning = torch.zeros([SIZE_1,SIZE_2,24])

                print('ICI1')

                lpc_coef_bidon = torch.tensor([-0.0642,   -0.9402,   -0.0657,    0.1920,
                                               -0.2137,   -0.0771,    0.1053,   -0.0500, 
                                               -0.0884,    0.0893,    0.1497,    0.0223,
                                               -0.0611,   -0.0665,    0.0337,    0.0246,
                                               -0.0202,   -0.0286,   -0.0552,   -0.0678,
                                                0.0454,    0.0757,    0.0714,    0.0100], dtype=torch.float)
                print('ICI2')
                for i in range(SIZE_1):
                    vocoder_conditioning[i,:] = lpc_coef_bidon
                    print('ICI3')

                vocoder_conditioning = vocoder_conditioning.contiguous().view(
                    SIZE_1, -1, 24
                )
                print('ICI4')

                if self.cuda:
                    prev_samples = prev_samples.cuda()
                    vocoder_conditioning = vocoder_conditioning.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, vocoder_conditioning, upper_tier_conditioning
                )
            print('mmmmmmmmmmmmmmmm')
            print(vocoder_conditioning.size())
            print(type(vocoder_conditioning))
            print(vocoder_conditioning)    
            print('mmmmmmmmmmmmmmmm')
            print(prev_samples.size())
            print(type(prev_samples))
            print(prev_samples)  
            print('mmmmmmmmmmmmmmmm')

            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size : i],
                volatile=True
            )
            print(prev_samples.size())
            print(type(prev_samples))
            print(prev_samples)  
            print('mmmmmmmmmmmmmmmm')
            if self.cuda:
                prev_samples = prev_samples.cuda()
                print(prev_samples.size())
                print(type(prev_samples))
                print(prev_samples)  
                print('mmmmmmmmmmmmmmmm')
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                                      .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]
