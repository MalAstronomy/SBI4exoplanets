import time 
start = time.time()

# import os
# os.environ["pRT_input_data_path"] = "/home/mvasist/pRT/input_data"

import numpy as np
import pylab as plt
import matplotlib.ticker as mticker
import h5py

import pymultinest

plt.rcParams['figure.figsize'] = (10, 6)
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc 
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.models import emission_model_diseq

from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.posteriors.ratio_based_posterior import RatioBasedPosterior
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
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Prior= utils.BoxUniform(low=torch.tensor([0.1, -1.5, -6.0, -3.5, -3.5,  2.0, 0.7, 300.0, 0., 0., 0.,\
                                  1., 0. ]), \
                      high=torch.tensor([1.6, 1.5, 3.0, 4.5, 4.5, 5.5, 2.0, 2300.0, 1., 1., 1.,\
                                      2., 1. ]), device='cuda')

def build_den():

    class VisualTrans(nn.Module):

        def __init__(self, file_path):
            super(VisualTrans, self).__init__()

            self.file_path = file_path

            self.model = ViT_modified(n_classes = 1,
                                image_size = (1, 962),  # image size is a tuple of (height, width)
                                patch_size = (1, 13),    # patch size is a tuple of (height, width)
                                dim = 16,
                                depth = 3,
                                heads = 16,
                                mlp_dim = 512,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )

            state_dict = torch.load(self.file_path, map_location='cpu')
            new_state_dict = OrderedDict()

            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                print('Ignoring test_dataset_size "' + str(e) + '"')

        def forward(self, inpt):
            theta, x = inpt
            theta = theta.unsqueeze_(1).unsqueeze_(1)
            x = x.unsqueeze_(1).unsqueeze_(1)
            x = torch.nn.functional.pad(x, (0, 2))
            inp = torch.cat((theta,x),3)

            out = self.model(inp)[0]  #another [0]- when the n=2
            return out
        
    return VisualTrans('/home/mvasist/scripts_new/model/model_vit.pth')

with h5py.File('/home/mvasist/scripts_new/datasets/dataset/_/test.h5', 'r') as f: 
    spec = torch.Tensor(f.get('spectra'))
    th = torch.Tensor(f.get('theta_reduced'))
#         l = torch.Tensor(f.get('label'))
    f.close()

#posterior = inference.build_posterior(build_den().to(device))
posterior = RatioBasedPosterior(method_family = 'snre_a', neural_net=build_den().to(device), \
                                prior= Prior, x_shape = torch.Size([1,947]))

observation = torch.load('/home/mvasist/scripts_new/observation/obs.pt') 

start = time.time()
sampls= 10 #200000

samples = posterior.sample((sampls,), x=observation)

end= time.time()
time_taken = (end-start)/3600  #hrs


log_probability = posterior.log_prob(samples, x= observation)


# # Saving the samples file

# df_samples = pd.DataFrame(samples.numpy())
# df_samples.to_csv('/home/mvasist/samples_new/...csv',mode='a', header=False)

# df_lnprob = pd.DataFrame(log_probability.numpy())
# df_lnprob.to_csv('/home/mvasist/samples_new/...csv',mode='a', header=False)