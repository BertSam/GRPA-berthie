# %%
import torch as t
import numpy as np

x = t.rand([6, 3])
X = t.rand(6,3)
print(x)
print(x.size())
print(type(x))
print('===')
print(X)
print(X.size())
print(type(X))


# y = t.tensor([2, 2])
 
# ##y.add_(x)

# #print(x)
# #print(y)

# #z = x + y 
# #z = t.mul(x, y)
# #print(z)


# # temp = x.view(-1, 2)
# # temp2 = x.permute(-1, 2)

# temp.size()



# m = t.nn.Conv1d(16, 33, 3, stride=2)
# input = t.randn(20, 16, 50)
# output = m(input)

# print("input")
# print(input.size())
# print("output")
# print(output.size())

# print(20*16*50)
# print(20*33*24)
# %%


default_params = {
    # model parameters
    'n_rnn': 1,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,

    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    'results_path': 'results',
    'epoch_limit': 1000,
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'cuda': True,
    'comet_key': None
}

params = dict(
        default_params)

print(params['n_rnn'])
print(params['dim'])

# %%
import numpy as np
frame_sizes = [160,16,2,2]

ns_frame_samples = map(int, np.cumprod(frame_sizes))

print(np.cumprod(frame_sizes))
print(list(ns_frame_samples))

x = zip(frame_sizes, ns_frame_samples)

print(tuple(x))

# %%
