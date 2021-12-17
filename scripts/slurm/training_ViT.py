import time 
start = time.time()

import os
# os.environ["pRT_input_data_path"] = "/home/mvasist/pRT/input_data"

import numpy as np
import pylab as plt
import matplotlib.ticker as mticker

import pymultinest

plt.rcParams['figure.figsize'] = (10, 6)
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc 
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.models import emission_model_diseq

from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.types import Array, OneOrMore, ScalarFloat

import pandas as pd
import sys
import csv
import h5py
import json
from pathlib import Path
import math
import glob
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms, models
from torchsummary import summary

from vit_pytorch.efficient import ViT
from linformer import Linformer
from vit_pytorch import ViT as ViT_modified

#files = glob.glob('/home/mvasist/scripts_new/datasets/dataset/_/onehot/*.h5')

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
    """
    def __init__(self, file_path, load_data, data_cache_size=4):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

        # Search for all h5 files
        p = Path(file_path)
        #print(p)
        assert(p.is_dir())
        
        files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index) #cache data
        x = torch.from_numpy(x)
        x = x.unsqueeze_(1)
        #x = transforms.Lambda(lambda h: h.repeat(1, 3, 1, 1))(x) #adding 3 channels 
        x = torch.nn.functional.pad(x, (0, 2, 0, 0)) #padding - 962 div by 13
        #x = x.view(-1,3,1,962)
        #print('get_item: ', x.size())

        # get label
        y = self.get_data("label", index) 
        y = torch.from_numpy(y)
        #y = y.view(-1,1)
        #print('len: ', y.size())
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all datasets, extracting them
            for dname, ds in h5_file.items():
                #print(dname, ds)
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(h5_file[dname][()], file_path)

                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': h5_file[dname][()].shape, 'cache_idx': idx})

    def _load_data(self, file_path): #into cache
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path,'r') as h5_file:
            for dname, ds in h5_file.items():
                #print(dname)
                idx = self._add_to_cache(h5_file[dname][()], file_path) #0 for data, 1 for labels

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx
                

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
            
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type
        

    def get_data(self, type, i):
        # This method loads the data in the file that the spectrum is in
        
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        #size of dataset in each file
        
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

#/home/mvasist/scripts_new/datasets/dataset/_/onehot/
dataset = HDF5Dataset('/scratch/mvasist/data/', load_data=False, data_cache_size=2)
# print(len(dataset))

split = [0.9, 0.1]
split_train = '0.9'
batch_size = 4 
indices = list(range(len(dataset)))
s = int(np.floor(split[1] * len(dataset)))

#shuffling
np.random.seed(111)
np.random.shuffle(indices)
train_indices, val_indices = indices[s:], indices[:s]
#print(train_indices, val_indices)
train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler)
val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=val_sampler)

dataloaders = {}
dataloaders['train'], dataloaders['val'] = train_dataloader, val_dataloader

def binary_acc(y_pred, y_test):
    #print(y_pred.argmax(axis=1), y_test.argmax(axis=1))
    correct_results_sum = (y_pred.argmax(axis=1) == y_test.argmax(axis=1)).sum().float()
    #print('crs', y_pred.argmax(axis=1) == y_test.argmax(axis=1), correct_results_sum, len(y_test))
    acc = correct_results_sum/len(y_test)
    acc = torch.round(acc * 100)
    
    return acc

dataset_size = len(dataset)

#962 = 13 * 74  - div into 74 patches 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT_modified(n_classes = 2,
                    image_size = (1, 962),  # image size is a tuple of (height, width)
                    patch_size = (1, 13),    # patch size is a tuple of (height, width)
                    dim = 16,
                    depth = 3,
                    heads = 16,
                    mlp_dim = 512,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

import gc
gc.collect()
torch.cuda.empty_cache()

#train

def train(n_epochs, model):
    
    best_loss = 0.0
    for epoch in range(n_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        for phase in ['train', 'val']:
            if phase == 'train':
                s = 0
                model.train()  # Set model to training mode
            else:
                s = 1
                model.eval()   # Set model to evaluate mode
            
            #both    
            running_loss = 0.0
            a = 0 
            for batch_idx, sample in enumerate(dataloaders[phase]):
                #print(batch_idx)
                inputs = sample[0].view(-1,1,1,962).to(device)
#                 print(inputs.element_size() * inputs.nelement() /1e6, 'MB')
                target = sample[1].view(-1,2).long().to(device)
#                 print(target.element_size() * target.nelement() /1e6, 'MB')
#                 print('boo', inputs.size(), target.size())
                optimizer.zero_grad()
#                 print('ok1')

                with torch.set_grad_enabled(phase == 'train'):
#                     print(np.shape(inputs))
                    output = model(inputs) #[None, ...]
                    #print('output', output)
                    #print(target.argmax(axis=1), target)
                    loss = criterion(torch.squeeze(output), torch.squeeze(target.argmax(axis=1)))
                    #print('loss: ', loss)
                    acc = binary_acc(torch.squeeze(output), torch.squeeze(target))
#                     print('acc done')
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   
#                         print('optimizer done')
                
                    running_loss += 1 * loss.item() * inputs.size(0) #loss for the phase/whole dataset
                
                if batch_idx % 1 == 0: 
                    #(batch_idx+1)*len(sample['input'])
                    a+= len(sample[0])
                    print('{} epoch: {} [{}/{} ({:0.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}'.format(phase,epoch,\
                            a,int(np.ceil(len(dataset)*split[s])),np.floor((100.*a)/(len(dataset)*split[s])), loss.item(), acc))
                                
            if phase == 'train':
                metrics[phase+'_loss'].append(running_loss/int(dataset_size*split[0]))
            else:
                metrics[phase+'_loss'].append(running_loss/int(dataset_size*split[1]))

            if phase == 'val': 
                if epoch ==  (n_epochs-1) or running_loss < best_loss:
                    print('saving')
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model_vit_sh.pth')
                    torch.save(model.state_dict(), model_path)
                    
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
                    
#         print('--------------------------------------------------------------------')
        

nb_epoch = 100
criterion = nn.CrossEntropyLoss() #nn.BCELoss()
# summary(model, (1, 1, 960))

optimizer = optim.Adam(model.parameters(), lr=float(0.001))

model_dir = '/home/mvasist/scripts_new/model/'
metrics_path = os.path.join(model_dir, 'metrics__vit_sh.json')

metrics = {
    'model': model_dir,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
#     'scheduler': scheduler.__class__.__name__,
    'dataset_size': int(len(dataset)),
    'train_size': int(split[0]*len(dataset)),
    'test_size': int(split[1]*len(dataset)),
    'n_epoch': nb_epoch,
    'batch_size': batch_size,
#     'learning_rate': [],
    'train_loss': [],
    'val_loss': []
}

train(nb_epoch, model)

end = time.time()

print('it takes: ', (end-start)/60, ' minutes')