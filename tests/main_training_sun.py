# encoding:  utf-8
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn 
from updators.state_updator import updator
from ops.sun_spin1d import SUNSpin1D, get_init_state
from algos.pesudocomplex_ppo import train 
import os
import argparse

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
Ops_args = dict(hamiltonian=SUNSpin1D, get_init_state=get_init_state, updator=updator)
Ham_args = dict(state_size=state_size, t=1, pbc=True)
net_args = dict(K=args.kernels, F=args.filters, layers=args.layers)
# input_fn = 'SUN_1d/save_model/model_499.pkl'
input_fn = 0
output_fn ='SUN_1d'

train(epochs=args.epochs, Ops_args=Ops_args, Ham_args=Ham_args, n_sample=args.n_sample, 
    n_optimize=args.n_optimize, learning_rate=args.lr, state_size=state_size, 
    save_freq=10, net_args=net_args, threads=args.threads, input_fn=input_fn, output_fn=output_fn)
