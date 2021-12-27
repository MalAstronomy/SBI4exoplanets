import time 
start = time.time()

import numpy as np
import pylab as plt
import matplotlib.ticker as mticker
import h5py
import glob

import pymultinest

plt.rcParams['figure.figsize'] = (10, 6)
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc 
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.models import emission_model_diseq

from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi, SNPE_A
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.types import Array, OneOrMore, ScalarFloat
import torch
import pandas as pd
import sys
import csv

from vit_pytorch.efficient import ViT
from linformer import Linformer
from vit_pytorch import ViT as ViT_modified
from collections import Counter, OrderedDict

import gc
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Prior= utils.BoxUniform(low=torch.tensor([0.1, -1.5, -6.0, -3.5, -3.5, 2.0, 0.7, 300.0, 0., 0., 0.,\
                                  1., 0. ]), \
                      high=torch.tensor([1.6, 1.5, 3.0, 4.5, 4.5, 5.5, 2.0, 2300.0, 1., 1., 1.,\
                                      2., 1. ]), device='cuda')

inference = SNPE_A(prior= Prior, device= 'cuda', classifier='mlp')

X=[]
T=[]
for i, file_path in enumerate(glob.iglob('/home/mvasist/scripts_new/datasets/dataset/_/onehot/*.h5')):
    if (i%1000 == 0):
        print(i)
    with h5py.File(file_path, 'r') as h5_file:
        spec = h5_file['data'][()]
        T.append(spec[:100, 0, :13])
        X.append(spec[:100, 0, 13:])
    if i==36000: break
            
comb_np_array_x = np.vstack(X)
x = torch.from_numpy(comb_np_array_x).type(torch.float32)
comb_np_array_T = np.vstack(T)
th_reduced = torch.from_numpy(comb_np_array_T).type(torch.float32)

x_norm = torch.nn.functional.normalize(x, dim=1)

inference = inference.append_simulations(th_reduced, x_norm)
# inference = inference.append_simulations(th_reduced.to(device), x.to(device))

density_estimator = inference.train()

posterior = inference.build_posterior(density_estimator)

observation = torch.load('/home/mvasist/scripts_new/observation/obs.pt') 

end = time.time()
print('time takes for loading: ', (end-start)/3600)

start = time.time()
sampls= 10000 #200000

samples = posterior.sample((sampls,), x=observation)
log_probability = posterior.log_prob(samples, x= observation)

end= time.time()
time_taken = (end-start)/3600  #hrs
print('time taken for sampling: ', (end-start)/3600)
# Saving the samples file

df_samples = pd.DataFrame(samples.numpy())
df_samples.to_csv('/home/mvasist/samples_new/samples_snpe_mlp__3_6MSim_10kSampl.csv',mode='a', header=False)

df_lnprob = pd.DataFrame(log_probability.numpy())
df_lnprob.to_csv('/home/mvasist/samples_new/lnprob_snpe_mlp__3_6MSim_10kSampl.csv',mode='a', header=False)
