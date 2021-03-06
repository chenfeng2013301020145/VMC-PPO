# encoding:  utf-8
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from updators.state_swap_updator import updator
from ops.HS_spin2d import J1J2_2DSquare, Heisenberg2DTriangle, get_init_state, value2onehot
from algos.complex_ppo import train
from algos.core import translation, identity, c6rotation, c4rotation, transpose, inverse
# from ops.operators import cal_op, Sz, Sx, SzSz
import os
import argparse
import scipy.io as sio
from utils_ppo import decimalToAny
import warnings

warnings.filterwarnings('ignore')

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
parser.add_argument('--dfs', type=float, default=0.01)
parser.add_argument('--warmup_length', type=int, default=500)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--warm_up_sample_length', type=int, default=20)
args = parser.parse_args()

state_size = [args.lattice_length, args.lattice_width, args.Dp]
TolSite = args.lattice_length*args.lattice_width
Ops_args = dict(hamiltonian=J1J2_2DSquare, get_init_state=get_init_state, updator=updator)
Ham_args = dict(state_size=state_size, pbc=True, j2=0.5)
net_args = dict(K=args.kernels, F=args.filters, relu_type='snake', pbc=True,
         sym_funcs=[c4rotation,transpose], momentum=[0,0], 
         apply_unique=False, MPphase=False, MPtype='NN', alpha=0.75)
# input_fn = 'HS_2d_sq_L4W4_vmcppo1/save_model/model_999.pkl'
input_fn = 0
output_fn ='HS_2d_j1j2_05_L'+str(args.lattice_length)+'W'+str(args.lattice_width)+'_vmcppo1'

trained_psi_model, state0, _ = train(epochs=args.epochs, Ops_args=Ops_args,
        Ham_args=Ham_args, n_sample=args.n_sample, n_optimize=args.n_optimize, seed=args.seed, preload_size=5000, batch_size=10000,
        learning_rate=args.lr, state_size=state_size, save_freq=10,
        net_args=net_args, threads=args.threads, input_fn=input_fn, load_state0=False, warmup_length=args.warmup_length,
        output_fn=output_fn, target_dfs=args.dfs, warm_up_sample_length=args.warm_up_sample_length)
# print(state0.shatarget_dfs
# calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model,
#             state0=state0, n_sample=5*args.n_sample, updator=updator,
#             get_init_state=get_init_state, threads=args.threads, batch_size=1000)
# '''
# sz, stdsz, IntCount = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))
# print([sz/TolSite, stdsz, IntCount])

# szsz, stdszsz, IntCount = calculate_op.get_value(operator=SzSz, op_args=dict(state_size=state_size, pbc=True))
# print([szsz/TolSite, stdsz, IntCount])

# sx, stdsx, IntCount = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))
# print([sx/TolSite, stdsx, IntCount])
# '''
# meane, stde, IntCount = calculate_op.get_value(operator=Heisenberg2DSquare, op_args=Ham_args)
# print([meane/TolSite, stde/TolSite, IntCount])

trained_psi_model.cpu()
# trained_theta_model.cpu()
def b_check():
    Dp = args.Dp
    L = args.lattice_length
    W = args.lattice_width
    N = L*W
    ms = 0 if L*W%2 == 0 else -0.5
    # chech the final distribution on all site configurations.
    basis_index = []
    basis_state = []

    for i in range(Dp**N):
        num_list = decimalToAny(i,Dp)
        state_v = np.array([0]*(N - len(num_list)) + num_list)
        if (state_v - (Dp-1)/2).sum() == ms:
            basis_index.append(i)
            basis_state.append(state_v.reshape(L,W))

    state_onehots = torch.zeros([len(basis_index), Dp, L, W], dtype=int)

    for i, state in enumerate(basis_state):
        state_onehots[i] = torch.from_numpy(value2onehot(state, Dp))

    # state_onehots[:,:,-1] = state_onehots[:,:,0]
    # state_onehots = state_onehots[spin_number.argsort(),:,:]
    # print(state_onehots.shape)

    psi = torch.squeeze(trained_psi_model(state_onehots.float())).detach().numpy()
    logphis = psi[:,0] - np.mean(psi[:,0])
    thetas = psi[:,1]
    probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
    print(np.sum(probs))

    # plt.figure(figsize=(8,6))
    # plt.bar(np.sort(spin_number), np.exp(logphis*2)/np.sum(np.exp(logphis*2)))
    # plt.show()

    sio.savemat('./data/test_data_HS2dtri_L'+str(args.lattice_length)
                +'W'+str(args.lattice_width)+'.mat',dict(probs=probs, logphis=logphis, thetas=thetas))

# b_check()
