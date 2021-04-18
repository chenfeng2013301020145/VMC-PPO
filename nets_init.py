# encoding: utf-8
# this file is to initialize NN with random wavefunction.

import numpy as np
import torch
import torch.nn as nn
from sampler.mcmc_sampler_complex_float import MCsampler
from core import mlp_cnn, get_paras_number, gradient
from utils import SampleBuffer, _get_unique_states, extract_weights, load_weights
import scipy.io as sio
import copy
import time
import os

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
# ------------------------------------------------------------------------
class train_Ops:
    def __init__(self, **kwargs):
        self._ham = kwargs.get('hamiltonian')
        self._get_init_state = kwargs.get('get_init_state')
        self._updator = kwargs.get('updator')
        # self._sampler = kwargs.get('sampler')
        
# ------------------------------------------------------------------------
def random_train(epochs=100, net_args=dict(), Ops_args=dict(), seed=0, batch_size=1000,
    Ham_args=dict(), learning_rate=1E-3, n_sample=1000, init_type='rand', threads=4,
    state_size=[10,10,2], load_state0=True, output_fn='test'):
    
    seed += 1000*np.sum(np.arange(threads))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = os.path.join('./results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator

    model = mlp_cnn(state_size=state_size, complex_nn=True, **net_args).to(gpu)
    mh_model = mlp_cnn(state_size=state_size, complex_nn=True, **net_args)
    print(model)
    print(get_paras_number(model))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.SmoothL1Loss()
    
    MHsampler.basis_warmup_sample = threads
    MHsampler._warmup = False
    MHsampler.accept = True
    # MHsampler.first_warmup()
    buffer = SampleBuffer(gpu)
    MHsampler._model.load_state_dict(model.state_dict())
    
    def get_fubini_study_distance(data):
        state, logphi0, theta0  = data['state'], data['logphi0'], data['theta0']
        
        psi = model(state)
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)
        
        phiold = torch.exp(2*logphi0[...,None])
        phinew = torch.exp(2*logphi)
        phiold_norm = phiold/phiold.sum()
        phinew_norm = phinew/phinew.sum()
        
        phi_ratio = (phinew_norm*phiold_norm.detach()).sqrt()
        deltatheta = theta - theta0[...,None].detach()
        
        phiold_phinew = (phi_ratio*torch.exp(1j*deltatheta)).sum()
        phinew_phiold = phiold_phinew.conj()
        
        #fsd = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold))
        fid = 1 - (phiold_phinew*phinew_phiold).abs()
        return fid
    
    # ------------------------------------------------------------------------
    # MHsampler._model.load_state_dict(small_model.state_dict())
    states, logphis, update_states, update_coeffs = MHsampler.parallel_mh_sampler()

    states, logphis, count, update_states, update_coeffs = _get_unique_states(states, logphis, update_states, update_coeffs)
    IntCount = len(states)

    #target_angle = np.random.uniform(-np.pi, np.pi, size=IntCount)
    #target_psi = np.random.normal(np.mean(logphis), size=IntCount)
    rand_psi = np.mean(logphis)*(np.random.rand(IntCount) + 1j*np.random.rand(IntCount))
    target_psi = torch.from_numpy(rand_psi).to(torch.cfloat)
    target_logphis = target_psi.abs().numpy()
    target_angle = target_psi.angle().numpy()
    # print(target_angle)
    buffer.update(states, target_logphis, target_angle, count, update_states, update_coeffs)

    tic = time.time()
    for i in range(epochs):
        data = buffer.get(batch_size=batch_size, batch_type='rand')
        state, logphi0, theta0  = data['state'], data['logphi0'], data['theta0']
        optimizer.zero_grad()

        #psi = model(state)
        #target_psi = torch.stack((logphi0, theta0), dim=1).detach()
        loss = get_fubini_study_distance(data)
        #loss = loss_func(psi, target_psi)
        #loss = MSE_loss(pred_logphis, logphis, pred_thetas, thetas, count)
        loss.backward()
        optimizer.step()

        if i%50 == 0:
            print('Epoch: {}, Loss: {:.5f}, IntCount: {}, Time: {}'
            .format(i, 1 - loss.data.item(), IntCount, time.time()-tic))

    # ------------------------------------------------------------------------
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_init.pkl'))
    return model.to(cpu)

if __name__ =='__main__':
    from updators.state_swap_updator import updator
    from operators.HS_spin2d import Heisenberg2DTriangle, get_init_state, value2onehot
    import torch.nn as nn
    from utils import decimalToAny

    state_size = [3, 2, 2]
    Ops_args = dict(hamiltonian=Heisenberg2DTriangle, get_init_state=get_init_state, updator=updator)
    Ham_args = dict(state_size=state_size, pbc=True)
    net_args = dict(K=2, F=[2,1], relu_type='softplus2', inverse_sym=False, bias=False)
    #input_mh_fn = 'HS_2d_tri_L3W4_SR/save_model/model_199.pkl'
    # input_fn = 'HS_2d_sq_L3W2/save_model/model_199.pkl'
    # output_fn ='HS_2d_tri_L3W2_vmcppo'
    output_fn = 'HS_2d_tri_L3W2_SRc'
    
    model = random_train(epochs=5000, net_args=net_args, Ops_args=Ops_args, seed=1183,
        Ham_args=Ham_args, learning_rate=3E-4, n_sample=70000, init_type='rand', threads=35,
        state_size=state_size, load_state0=False, output_fn=output_fn)
    
    def b_check():
        Dp = state_size[-1]
        L = state_size[0]
        W = state_size[1]
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

        psi = torch.squeeze(model(state_onehots.float())).detach().numpy()
        logphis = psi[:,0]
        thetas = psi[:,1]
        probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
        print(np.sum(probs))

        sio.savemat('./data/test_data_HS2dtri_L3W2init.mat',dict(probs=probs, logphis=logphis, thetas=thetas))

    b_check()
