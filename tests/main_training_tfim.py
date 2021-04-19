# encoding:  utf-8
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn 
from updators.state_flip_updator import updator
from ops.tfim_spin1d import TFIMSpin1D, get_init_state
from algos.pesudocomplex_ppo import train 
from ops.operators import cal_op, Sz, Sx, SzSz
import os
import argparse
import scipy.io as sio
from utils import decimalToAny

'''
# ----------------------- test ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--n_sample', type=int, default=1000)
parser.add_argument('--n_optimize', type=int, default=10)
parser.add_argument('--lr',type=float, default=1E-3)
parser.add_argument('--lattice_size',type=int, default=10)
parser.add_argument('--Dp', type=int, default=2)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--kernels', type=int, default=3)
parser.add_argument('--filters', type=int, default=4)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--g', type=float, default=1.)
args = parser.parse_args()

state_size = [args.lattice_size, args.Dp]
Ops_args = dict(hamiltonian=TFIMSpin1D, get_init_state=get_init_state, updator=updator)
Ham_args = dict(g=args.g, state_size=state_size, pbc=True)
net_args = dict(K=args.kernels, F=args.filters, layers=args.layers)
input_fn = 0
output_fn ='TFIM_1d_L16'

trained_psi_model, state0, _ = train(epochs=args.epochs, Ops_args=Ops_args,
         Ham_args=Ham_args, n_sample=args.n_sample, 
        n_optimize=args.n_optimize, learning_rate=args.lr, state_size=state_size, 
        save_freq=10, dimensions='1d', net_args=net_args, input_fn=input_fn, load_state0=False,
        threads=args.threads, output_fn=output_fn, sample_division=5)

calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model, state0=state0, n_sample=args.n_sample, 
            updator=updator, get_init_state=get_init_state, threads=args.threads, sample_division=20)


sz, stdsz, IntCount = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))
print([sz/args.lattice_size, stdsz, IntCount])

szsz, stdszsz, IntCount = calculate_op.get_value(operator=SzSz, op_args=dict(state_size=state_size,pbc=True))
print([szsz/args.lattice_size, stdsz, IntCount])

sx, stdsx, IntCount = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))
print([sx/args.lattice_size, stdsx, IntCount])

meane, stde, IntCount = calculate_op.get_value(operator=TFIMSpin1D, op_args=Ham_args)
print([meane/args.lattice_size, stde, IntCount])
'''
'''
def b_check():
    Dp = args.Dp
    N = args.lattice_size
    # chech the final distribution on all site configurations.
    rang = range(Dp**N)
    state_onehots = torch.zeros([len(rang), Dp, N], dtype=int)
    spin_number = np.zeros([len(rang)])

    for i in rang:
        num_list = decimalToAny(i,Dp)
        state_v = np.array([0]*(N - len(num_list)) + num_list)
        state_onehots[i, state_v, range(N)] = 1
        spin_number[i] = np.sum(state_v)
    
    # state_onehots[:,:,-1] = state_onehots[:,:,0]
    # state_onehots = state_onehots[spin_number.argsort(),:,:]

    psis = torch.squeeze(trained_model(state_onehots.float())).detach().numpy()
    logphis = psis[:,0]
    probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
    print(np.sum(probs))
    
    
    # plt.figure(figsize=(8,6))
    # plt.bar(np.sort(spin_number), np.exp(logphis*2)/np.sum(np.exp(logphis*2)))
    # plt.show()
    
    sio.savemat('test_data.mat',dict(spin_number=spin_number, probs=probs))
'''
# b_check()
# phase diagram for g \in [-2,2]

sz_list = []
sx_list = []
energy = []
state_size = [16,2]
Ops_args = dict(hamiltonian=TFIMSpin1D, get_init_state=get_init_state, updator=updator)
net_args = dict(K=2, F=4, layers=2)
output_fn ='TFIM_1d'
for g in np.linspace(-1,1,30):
    Ham_args = dict(state_size=state_size,g=g, pbc=True)
    trained_psi_model, state0, mean_e = train(epochs=50, Ops_args=Ops_args, Ham_args=Ham_args, 
            n_sample=45000, n_optimize=100, learning_rate=3E-4, state_size=state_size, dimensions='1d',
            save_freq=10, net_args=net_args, threads=45, output_fn=output_fn, sample_division=5)
    
    calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model, state0=state0, n_sample=45000, 
            updator=updator, get_init_state=get_init_state, threads=45, sample_division=10)
    
    sz, _, _ = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))

    sx, _, _ = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))

    print([sz, sx])
    sz_list.append(sz)
    sx_list.append(sx)
    energy.append(mean_e)

sio.savemat('tfim1d_pd_data.mat',{'sz':np.array(sz_list), 'sx':np.array(sx_list), 'energy':np.array(energy)})
