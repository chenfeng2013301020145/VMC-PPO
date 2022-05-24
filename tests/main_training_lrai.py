# encoding:  utf-8
from ast import parse
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from updators.state_flip_updator import updator
from ops.tfim_spin1d import LRAISpin1D, get_init_state
from algos.complex_ppo import train
from algos.core import inverse
from ops.operators_v1 import cal_op, Sz, Sx, SzSz
import os
import argparse
import scipy.io as sio
from utils_ppo import decimalToAny

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
parser.add_argument('--kernels', nargs='+', type=int, default=[3])
parser.add_argument('--filters', nargs='+', type=int, default=[4,3,2])
parser.add_argument('--g', type=float, default=1.)
parser.add_argument('--alpha', type=float, default=3)
parser.add_argument('--dfs', type=float, default=0.01)
parser.add_argument('--warmup_length', type=int, default=500)
args = parser.parse_args()

state_size = [args.lattice_size, args.Dp]
Ops_args = dict(hamiltonian=LRAISpin1D, get_init_state=get_init_state, updator=updator)
Ham_args = dict(g=args.g, alpha=args.alpha, state_size=state_size, pbc=True)
net_args = dict(K=args.kernels, F=args.filters, pbc=True, complex_nn=True)
input_fn = 0
output_fn ='TFIM_1d_L'+str(args.lattice_size)+'_vmcppo1'

trained_psi_model, state0, _ = train(epochs=args.epochs, Ops_args=Ops_args,
         Ham_args=Ham_args, n_sample=args.n_sample, seed=12386, batch_size=5000,
        n_optimize=args.n_optimize, learning_rate=args.lr, state_size=state_size,
        save_freq=10, dimensions='1d', net_args=net_args, input_fn=input_fn, load_state0=False,
        threads=args.threads, output_fn=output_fn, warmup_length=args.warmup_length)

calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model, state0=state0, n_sample=5*args.n_sample,
            updator=updator, get_init_state=get_init_state, threads=args.threads, batch_size=5000)


sz, stdsz, IntCount = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))
print([sz/args.lattice_size, stdsz, IntCount])

szsz, stdszsz, IntCount = calculate_op.get_value(operator=SzSz, op_args=dict(state_size=state_size,pbc=True))
print([szsz/args.lattice_size, stdsz, IntCount])

sx, stdsx, IntCount = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))
print([sx/args.lattice_size, stdsx, IntCount])

meane, stde, IntCount = calculate_op.get_value(operator=LRAISpin1D, op_args=Ham_args)
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
'''
sz_list = []
sx_list = []
energy = []
state_size = [16,2]
Ops_args = dict(hamiltonian=LRAISpin1D, get_init_state=get_init_state, updator=updator)
net_args = dict(K=[12], F=[8,6,4], complex_nn=True, pbc=True)
output_fn ='LRAI_N16_1d_vmcppo1'
cnt = 0
for g in np.linspace(-2,0,40):
    if cnt == 0:
        input_fn = 0
        load_state0=False
        warmup_epochs=10
    else:
        input_fn = 'LRAI_N16_1d_vmcppo1/save_model/model_'+str(fn_num)+'.pkl'
        load_state0 = True
        warmup_epochs=1

    Ham_args = dict(state_size=state_size,g=g, pbc=True, alpha=0.5)
    trained_psi_model, state0, mean_e, fn_num = train(epochs=500, Ops_args=Ops_args, Ham_args=Ham_args,
            n_sample=1000, n_optimize=100, learning_rate=1E-4, state_size=state_size, dimensions='1d',
            save_freq=10, net_args=net_args, threads=24, output_fn=output_fn, input_fn=input_fn, load_state0=load_state0,
            batch_size=5000, seed=12386, warmup_length=500, warmup_epochs=warmup_epochs)

    calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model, state0=state0, n_sample=5000,
            updator=updator, get_init_state=get_init_state, threads=24, batch_size=5000)

    sz, _, _ = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))

    sx, _, _ = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))

    print([sz, sx])
    sz_list.append(sz)
    sx_list.append(sx)
    energy.append(mean_e)
    cnt += 1

sio.savemat('lrai1dk12_N16pbc_alpha05_pd_data.mat',{'sz':np.array(sz_list), 
        'sx':np.array(sx_list), 'energy':np.array(energy)})
'''
hc = []

for alpha in np.linspace(0.1,3,20):
    sz_list = []
    sx_list = []
    energy = []
    state_size = [24,2]
    Ops_args = dict(hamiltonian=LRAISpin1D, get_init_state=get_init_state, updator=updator)
    net_args = dict(K=[12], F=[10,8,6,4], complex_nn=True, pbc=True)
    output_fn ='LRAI_N16_1d_vmcppo1'
    cnt = 0
    for g in np.linspace(-1.5,0,50):
        if cnt == 0:
            input_fn = 0
            load_state0=False
            warmup_epochs=10
        else:
            input_fn = 'LRAI_N16_1d_vmcppo1/save_model/model_'+str(fn_num)+'.pkl'
            load_state0 = True
            warmup_epochs=1

        Ham_args = dict(state_size=state_size,g=g, pbc=True, alpha=alpha)
        trained_psi_model, state0, mean_e, fn_num = train(epochs=500, Ops_args=Ops_args, Ham_args=Ham_args,
                n_sample=500, n_optimize=80, learning_rate=1E-4, state_size=state_size, dimensions='1d',
                save_freq=10, net_args=net_args, threads=48, output_fn=output_fn, input_fn=input_fn, load_state0=load_state0,
                batch_size=15000, seed=12386, warmup_length=1000, warmup_epochs=warmup_epochs)

        calculate_op = cal_op(state_size=state_size, psi_model=trained_psi_model, state0=state0, n_sample=5000,
                updator=updator, get_init_state=get_init_state, threads=48, batch_size=15000)

        sz, _, _ = calculate_op.get_value(operator=Sz, op_args=dict(state_size=state_size))

        sx, _, _ = calculate_op.get_value(operator=Sx, op_args=dict(state_size=state_size))

        print([sz, sx])
        sz_list.append(sz)
        sx_list.append(sx)
        energy.append(mean_e)
        cnt += 1

    sio.savemat('./data/lrai_N24_data/lrai1dk12_N24pbc_alpha'+str(alpha)+'_pd_data.mat',{'sz':np.array(sz_list), 
            'sx':np.array(sx_list), 'energy':np.array(energy)})

    hc_label = np.argmin(np.gradient(np.array(sx_list)))
    h = np.linspace(-1.5,0,50)
    hc.append(h[hc_label])

sio.savemat('./data/lrai_N16_data/lrai1dk12_N24pbc_hc_data.mat',{'hc':np.array(hc)})
