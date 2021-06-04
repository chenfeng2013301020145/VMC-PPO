import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from updators.state_swap_updator import updator
from ops.HS_spin2d import value2onehot
from algos.core import mlp_cnn, mlp_cnn_sym, get_paras_number, gradient
from algos.core import translation, reflection, identity, c2rotation, transpose2, inverse
from ops.operators import cal_op, Sz, Sx, SzSz
import os
import argparse
import scipy.io as sio
from utils import decimalToAny

# ----------------------- test ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--lattice_length',type=int, default=10)
parser.add_argument('--lattice_width',type=int, default=10)
parser.add_argument('--Dp', type=int, default=2)
parser.add_argument('--kernels', nargs='+', type=int, default=[3])
parser.add_argument('--filters', nargs='+', type=int, default=[4, 3, 2])
parser.add_argument('--model_id', type=int, default=99)
args = parser.parse_args()

state_size = [args.lattice_length, args.lattice_width, args.Dp]
TolSite = args.lattice_length*args.lattice_width
net_args = dict(K=args.kernels, F=args.filters, relu_type='selu', sym_funcs=[c2rotation, transpose2, inverse], momentum=[0,0])
input_fn = 'HS_2d_tri_L'+str(args.lattice_length)+'W'+str(args.lattice_width)+'_vmcppo/save_model/model_'+str(args.model_id)+'.pkl'

psi_model, _ =  mlp_cnn_sym(state_size=state_size, complex_nn=True, **net_args)
load_model = torch.load(os.path.join('../results', input_fn))
psi_model.load_state_dict(load_model)

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

    psi = torch.squeeze(psi_model(state_onehots.float())).detach().numpy()
    logphis = psi[:,0] - np.mean(psi[:,0])
    thetas = psi[:,1]
    probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
    print(np.sum(probs))

    # plt.figure(figsize=(8,6))
    # plt.bar(np.sort(spin_number), np.exp(logphis*2)/np.sum(np.exp(logphis*2)))
    # plt.show()

    sio.savemat('./data/test_data_HS2dtri_L'+str(args.lattice_length)
                +'W'+str(args.lattice_width)+'.mat',dict(probs=probs, logphis=logphis, thetas=thetas))

b_check()