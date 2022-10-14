# encoding:  utf-8
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
from updators.state_swap_updator import updator
from ops.HS_spin2d import J1J2_2DSquare, get_init_state
from algos.complex_ppo2 import train
from algos.core_gcnn2 import identity
import warnings

warnings.filterwarnings('ignore')

# ----------------------- test ----------------------
state_size = [6, 6, 2]
TolSite = 36
Ops_args = dict(hamiltonian=J1J2_2DSquare, get_init_state=get_init_state, updator=updator)
Ham_args = dict(state_size=state_size, pbc=True, j2=0.5, Marshall_sign=False)
net_args = dict(K=[5], F=[4,4,4,4,4], pbc=True,
                sym_funcs=[identity], alpha=0, custom_filter=False, lattice_shape='sq')
#input_fn = 'HS_2d_j1j2_05_gcnnT6_L8W8_ppo/save_model/model_200.pkl'
input_fn = 0
output_fn ='HS_2d_j1j2_05_gcnnD1_L6W6_ppo'

trained_psi_model, state0 = train(epochs=1000, Ops_args=Ops_args,
        Ham_args=Ham_args, n_sample=500, n_optimize=50, seed=316890, preload_size=50000, batch_size=10000,
        learning_rate=2e-4, state_size=state_size, save_freq=10,
        net_args=net_args, threads=20, input_fn=input_fn, load_state0=False, warmup_length=300,
        output_fn=output_fn, target_dfs=2, max_beta=10, min_beta=0.5,
        warm_up_sample_length=1, verbose=False, phase_constriction=True, precision=torch.float64)