import time 
start = time.time()

# import os
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

import torch
import pandas as pd
import sys
import csv

# Load scattering version of pRT
atmosphere = Radtrans(line_species = ['H2O',
                                      'CO_all_iso',
                                      'CH4',
                                      'NH3',
                                      'CO2',
                                      'H2S',
                                      'VO',
                                      'TiO_all_Exomol', 
#                                       'FeH',
                                      'PH3',
                                      'Na_allard',
                                      'K_allard'],
                      cloud_species = ['MgSiO3(c)_cd',"Fe(c)_cd"],
                      rayleigh_species = ['H2', 'He'],
                      continuum_opacities = ['H2-H2', 'H2-He'],
                      wlen_bords_micron = [0.95, 2.45],
                      do_scat_emis = True)

pressures = np.logspace(-6, 2, 154)
atmosphere.setup_opa_structure(pressures)

def Simulator(params): 

    '''

    Dictionary of required parameters:
                    *  D_pl : Distance to the planet in [cm]
                    *  log_g : Log of surface gravity
                    *  R_pl : planet radius [cm]
                    *  T_int : Interior temperature of the planet [K]
                    *  T3 : Innermost temperature spline [K]
                    *  T2 : Middle temperature spline [K]
                    *  T1 : Outer temperature spline [K]
                    *  alpha : power law index in tau = delta * press_cgs**alpha
                    *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                    *  Fe/H : Metallicity
                    *  C/O : Carbon to oxygen ratio
                    *  log_kzz : Vertical mixing parameter
                    *  fsed : sedimentation parameter
                    *  log_X_cb : Scaling factor for equilibrium cloud abundances.
                    
                    
    Parameter  Value         Parameter                     Value
    T 1        330.6 K       log(X_0 Fe /X_eq Fe)          -0.86
    T 2        484.7 K       log(X_0 MgSiO3 /X_eq MgSiO3)  -0.65
    T 3        687.6 K       fsed                           3
    log(δ)     -7.51         log(K zz /cm 2 s −1)           8.5
    α          1.39          σg                             2
    T 0        1063.6 K      R_P                            1 R J
    C/O        0.55          log(g/cm s −2)                 3.75
    [Fe/H]     0             log(P quench/bar)             -10 

    C/O, Fe/H, log_pquench, XFe, XMgSiO3, fsed, log_kzz, sigma_lnorm, log_g, R_pl,
    T_int, T3, T2, T1, alpha, log_delta- Molliere

    C/O, Fe/H, log_pquench, XFe, XMgSiO3, log_g, R_pl,
    T_int, T3, T2, T1, alpha, log_delta - me

    '''

    #16 params for simulation.
    
    #Maybe I need to change the params from numpy/float to torch 
    
    CO = params[0].numpy()                 # 0.55
    FeH = params[1].numpy()                # 0.
    log_pquench = params[2].numpy()        # -10.
    XFe = params[3].numpy()                # -0.86
    XMgSiO3 = params[4].numpy()            # -0.65
    fsed = params[5].numpy()               # 3. 
    log_kzz = params[6].numpy()            # 8.5
    sigma_lnorm = params[7].numpy()        # 2.
    log_g = params[8].numpy()              # 3.75
    R_pl = params[9].numpy()               # 1
    T_int = params[10].numpy()             # 1063.6
    T3 = params[11].numpy()                # 687.6 
    T2 = params[12].numpy()                # 484.7 
    T1 = params[13].numpy()                # 330.6 
    alpha = params[14].numpy()             # 1.39
    log_delta = params[15].numpy()         # -7.51
    
    # print(CO, FeH,log_pquench, XFe, XMgSiO3, fsed, log_kzz, sigma_lnorm, log_g, R_pl, T_int, T3, T2, T1, \
        #   alpha, log_delta)
    
    parameters={}
    parameters['D_pl'] = Parameter(name = 'D_pl', is_free_parameter = False, value = 41.2925*nc.pc) 
    parameters['log_g'] = Parameter(name ='log_g',is_free_parameter = False, value = log_g)
    parameters['R_pl'] = Parameter(name = 'R_pl', is_free_parameter = False, value = R_pl* nc.r_jup_mean)
    parameters['T_int'] = Parameter(name ='T_int',is_free_parameter = False, value = T_int)
    parameters['T3'] = Parameter(name = 'T3', is_free_parameter = False, value = T3)
    parameters['T2'] = Parameter(name ='T2',is_free_parameter = False, value = T2)
    parameters['T1'] = Parameter(name = 'T1', is_free_parameter = False, value = T1)
    parameters['alpha'] = Parameter(name ='alpha',is_free_parameter = False, value = alpha)
    parameters['log_delta'] = Parameter(name ='log_delta',is_free_parameter = False, value = log_delta)
    parameters['sigma_lnorm'] = Parameter(name ='sigma_lnorm',is_free_parameter = False, value = sigma_lnorm)
    parameters['log_pquench'] = Parameter(name ='log_pquench',is_free_parameter = False, value = log_pquench)
    parameters['Fe/H'] = Parameter(name ='Fe/H',is_free_parameter = False, value = FeH)
    parameters['C/O'] = Parameter(name ='C/O',is_free_parameter = False, value = CO)
    parameters['log_kzz'] = Parameter(name ='log_kzz',is_free_parameter = False, value = log_kzz)
    parameters['fsed'] = Parameter(name ='fsed',is_free_parameter = False, value = fsed)
    parameters['log_X_cb'+ '_Fe(c)'] = Parameter(name ='log_X_cb'+'_Fe(c)',is_free_parameter = False, value = XFe)
    parameters['log_X_cb'+'_MgSiO3(c)'] = Parameter(name ='log_X_cb'+'_MgSiO3(c)',is_free_parameter = False, value = XMgSiO3)
    parameters['pressure_scaling'] = Parameter(name ='pressure_scaling',is_free_parameter = False, value = 10)
    parameters['pressure_width'] = Parameter(name ='pressure_width',is_free_parameter = False, value = 3)
    parameters['pressure_simple'] = Parameter(name ='pressure_simple',is_free_parameter = False, value = 100)
    
    wlen, flux , p, t = emission_model_diseq(atmosphere, parameters, AMR = True)
    
    p_tensor = torch.Tensor(p)
    t_tensor = torch.Tensor(t)
    p_tensor = p_tensor.unsqueeze(0)
    t_tensor = t_tensor.unsqueeze(0)
    pt = torch.cat((p_tensor, t_tensor), 0)

    flux_tensor = torch.Tensor(flux)    
    flux_tensor_ = flux_tensor.unsqueeze(0)
    
    pres = len(p)
    pd = (0, flux_tensor.size()[0]-pres) 
    pt_padded = torch.nn.functional.pad(pt, pd, "constant", 0)
    ptf = torch.cat((pt_padded,flux_tensor_),0)                #ptf[0][:pres] = p, ptf[1][:pres] = t, ptf[2] = f
    # print(ptf.size())

    return  ptf
    
Prior= utils.BoxUniform(low=torch.tensor([0.1, -1.5, -6.0, -3.5, -3.5, 1.0, 5.0, 1.05, 2.0, 0.7, 300.0, 0., 0., 0.,\
                                  1., 0. ]), \
                      high=torch.tensor([1.6, 1.5, 3.0, 4.5, 4.5, 11.0, 13.0, 3.0, 5.5, 2.0, 2300.0, 1., 1., 1.,\
                                      2., 1. ]))


sim = 5000 #Run 1000 such in parallel (This amounts to a 5M sim spec)

simulator, prior = prepare_for_sbi(Simulator, Prior)

inference = SNRE_A(prior= prior, device= 'cpu')

#records every 100 sim
for i in range(0, int(sim/100)):
    theta, ptx = simulate_for_sbi(simulator, proposal=prior, num_simulations= 100) 
    
    theta_np = theta.numpy()
    T = pd.DataFrame(theta_np)
    T.to_csv('/home/mvasist/simulations_new/16_params/T_5Msim_'+ str(sys.argv[1]) + '.csv',mode='a', header=False)
    
    ptx_np = ptx.numpy()                   # Tn = ptx[n][947:154+947]
    ptX = pd.DataFrame(ptx_np)               # Pn = ptx[n][:154]     # Xn = ptx[n][947*2:]                     
    ptX.to_csv('/home/mvasist/simulations_new/16_params/X_5Msim_'+ str(sys.argv[1]) + '.csv',mode='a', header=False)                                             
                                                                       
end = time.time()
time_taken = (end-start)/3600   #hrs

with open('../time/16params/time_5Msim_'+str(sys.argv[1])+'.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the data
    writer.writerow([time_taken])


