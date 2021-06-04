# encoding:  utf-8
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from updators.state_kagome_swap_updator import updator
from ops.HS_spin2d_Kagome import Heisenberg2DKagome, get_init_state, value2onehot
from ops.HS_spin2d_Kagome import state2_to_state3, state3_to_state2
from algos.complex_ppo import train
from algos.core import reflection, identity, c4rotation
from ops.operators import cal_op, Sz, Sx, SzSz
import os
import argparse
import scipy.io as sio
from utils import decimalToAny


# ----------------------- test ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--n_sample', type=int, default=1000)
parser.add_argument('--n_optimize', type=int, default=10)
parser.add_argument('--lr',type=float, default=1E-3)
parser.add_argument('--lattice_length',type=int, default=10)
parser.add_argument('--lattice_width',type=int, default=10)
parser.add_argument('--Dp', type=int, default=2)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--kernels', nargs='+', type=int, default=[3])
parser.add_argument('--filters', nargs='+', type=int, default=[4, 3, 2])
parser.add_argument('--dfs', type=float, default=10)
args = parser.parse_args()

state_size = [args.lattice_length, args.lattice_width, args.Dp]
TolSite = (state_size[0]//2)*(state_size[1]//2)*3
Ops_args = dict(hamiltonian=Heisenberg2DKagome, get_init_state=get_init_state, updator=updator)
Ham_args = dict(state_size=state_size, pbc=True)
net_args = dict(K=args.kernels, F=args.filters, relu_type='selu', sym_funcs=[identity], stride0=[2], momentum=[0,0])
# input_fn = 'HS_2d_tri_L4W2/save_model/model_99.pkl'
input_fn = 0
output_fn ='HS_2d_Ka_L'+str(args.lattice_length)+'W'+str(args.lattice_width)+'_vmcppo'

trained_psi_model, state0, _ = train(epochs=args.epochs, Ops_args=Ops_args,
        Ham_args=Ham_args, n_sample=args.n_sample, n_optimize=args.n_optimize, seed = 2867,
        learning_rate=args.lr, state_size=state_size, save_freq=10, dimensions='2d',
        net_args=net_args, threads=args.threads, input_fn=input_fn, 
        output_fn=output_fn, target_dfs=args.dfs, sample_division=5, 
        TolSite=TolSite)
# print(state0.shape)
calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model,
            state0=state0, n_sample=args.n_sample, updator=updator,
            get_init_state=get_init_state, threads=args.threads, sample_division=10)
'''
sz, stdsz, IntCount = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))
print([sz/TolSite, stdsz, IntCount])

szsz, stdszsz, IntCount = calculate_op.get_value(operator=SzSz, op_args=dict(state_size=state_size, pbc=True))
print([szsz/TolSite, stdsz, IntCount])

sx, stdsx, IntCount = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))
print([sx/TolSite, stdsx, IntCount])
'''
meane, stde, IntCount = calculate_op.get_value(operator=Heisenberg2DKagome, op_args=Ham_args)
print([meane/TolSite, stde/TolSite, IntCount])

trained_psi_model.cpu()
# trained_theta_model.cpu()
def b_check():
    Dp = args.Dp
    L = args.lattice_length
    W = args.lattice_width
    L3 = L // 2
    W3 = W // 2 
    num = L3*W3*3
    ms = 0 if num%2 == 0 else -0.5
    # chech the final distribution on all site configurations.
    basis_index = []
    basis_state = []

    for i in range(Dp**num):
        num_list = decimalToAny(i,Dp)
        state_v = np.array([0]*(num - len(num_list)) + num_list)
        if (state_v - (Dp-1)/2).sum() == ms:
            basis_index.append(i)
            basis_state.append(state_v.reshape([L3,W3,3]))

    state_onehots = torch.zeros([len(basis_index), Dp, L, W], dtype=int)

    for i, state3 in enumerate(basis_state):
        state2 = state3_to_state2(value2onehot(state3, Dp), state_size)
        state_onehots[i] = torch.from_numpy(state2)

    # state_onehots[:,:,-1] = state_onehots[:,:,0]
    # state_onehots = state_onehots[spin_number.argsort(),:,:]

    psi = torch.squeeze(trained_psi_model(state_onehots.float())).detach().numpy()
    logphis = psi[:,0] - np.mean(psi[:,0])
    thetas = psi[:,1]
    probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
    print(np.sum(probs))

    sio.savemat('./data/test_data_HS2dKa_L'+str(args.lattice_length)
                +'W'+str(args.lattice_width)+'.mat',dict(probs=probs, logphis=logphis, thetas=thetas))

b_check()
