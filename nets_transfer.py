# encoding: utf-8
# this file is to transfer a wavefuntion from a small network to a bigger one.

import numpy as np
import torch
import torch.nn as nn
from sampler.mcmc_sampler_complex import MCsampler
from algos.core import mlp_cnn, get_paras_number, gradient
from utils import SampleBuffer, get_logger, _get_unique_states
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
def transfer(epochs=100, small_net_args=dict(), big_net_args=dict(), Ops_args=dict(),
    Ham_args=dict(), learning_rate=1E-3, n_sample=1000, init_type='rand', threads=4,
    batch_size = 1000, state_size=[10,10,2], input_mh_fn=0, load_state0=True, 
    output_fn='test'):

    output_dir = os.path.join('./results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator

    small_model = mlp_cnn(state_size=state_size, **small_net_args).to(gpu)
    mh_model = mlp_cnn(state_size=state_size, **small_net_args)
    print('original_model:')
    print(mh_model)
    print(get_paras_number(mh_model))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)
    if input_mh_fn != 0:
        load_model = torch.load(os.path.join('./results', input_mh_fn))
        small_model.load_state_dict(load_model)
        # theta_model.load_state_dict(load_models['theta_model']) 
        if load_state0:
            fn_name = os.path.split(os.path.split(input_mh_fn)[0])
            mat_content = sio.loadmat(os.path.join('./results',fn_name[0], 'state0.mat'))
            MHsampler.single_state0 = mat_content['state0']

    big_model = mlp_cnn(state_size=state_size, **big_net_args).to(gpu)
    print('transferred_to:')
    print(big_model)
    print(get_paras_number(big_model))

    buffer = SampleBuffer(gpu)
    optimizer = torch.optim.Adam(big_model.parameters(), lr=learning_rate)
    # loss_func = torch.nn.SmoothL1Loss()
    
    def get_fubini_study_distance(data):
        state, count, logphi0, theta0  = data['state'], data['count'], data['logphi0'], data['theta0']

        psi = big_model(state.float())
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)
        
        deltalogphi = logphi - logphi0[...,None].detach()
        deltalogphi = deltalogphi - deltalogphi.mean()
        deltatheta = theta - theta0[...,None].detach()
        
        phiold_phinew = (count[...,None]*torch.exp(deltalogphi)*torch.exp(1j*deltatheta)).sum()
        phinew_phiold = phiold_phinew.conj()
        phiold_phiold = count.sum()
        phinew_phinew = (count[...,None]*torch.exp(2*deltalogphi)).sum()
        
        fsd = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew))
        return fsd.abs()

    MHsampler._model.load_state_dict(small_model.state_dict())
    MHsampler.first_warmup()
    MHsampler.basis_warmup_sample = threads
    # ------------------------------------------------------------------------
    for epoch in range(epochs):
        # MHsampler._model.load_state_dict(small_model.state_dict())
        sample_tic = time.time()
        states, logphis, update_states, update_coeffs = MHsampler.parallel_mh_sampler()

        states, _, counts, update_states, update_coeffs = _get_unique_states(states, logphis,
                                                                            update_states, update_coeffs)
        psi = small_model(torch.from_numpy(states).float().to(gpu))
        logphis = psi[:, 0].reshape(len(states)).cpu().detach().numpy()
        thetas = psi[:, 1].reshape(len(states)).cpu().detach().numpy()
        buffer.update(states, logphis, thetas, counts, update_states, update_coeffs)
        sample_toc = time.time()
        IntCount = len(states)

        for _ in range(100):
            data = buffer.get(batch_type='rand',batch_size=batch_size)
            optimizer.zero_grad()
            
            loss = get_fubini_study_distance(data)
            #loss = loss_func(pred_psi, psi)
            # loss = MSE_loss(pred_logphis, logphis, pred_thetas, thetas, count)
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Loss: {:.5f}, IntCount: {}, Sample_time: {:.5f}'.
            format(epoch, loss.data.item(), IntCount, sample_toc-sample_tic))

    # ------------------------------------------------------------------------
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(big_model.state_dict(), os.path.join(save_dir, 'model_TL.pkl'))
    sio.savemat(os.path.join(output_dir, 'state0.mat'), {'state0': MHsampler.single_state0})
    return big_model.to(cpu)

if __name__ =='__main__':
    from updators.state_swap_updator import updator
    from ops.HS_spin2d import Heisenberg2DTriangle, get_init_state, value2onehot
    import torch.nn as nn
    from utils import decimalToAny

    state_size = [3, 4, 2]
    Ops_args = dict(hamiltonian=Heisenberg2DTriangle, get_init_state=get_init_state, updator=updator)
    Ham_args = dict(state_size=state_size, pbc=True)
    net_args = dict(K=3, F=[5,4,3], complex_nn=True, relu_type='softplus2')
    mh_net_args = dict(K=3, F=[4,2], complex_nn=True, relu_type='softplus2')
    input_mh_fn = 'HS_2d_tri_L3W3_vmcppo/save_model/model_499.pkl'
    # input_fn = 'HS_2d_sq_L3W2/save_model/model_199.pkl'
    output_fn ='HS_2d_tri_L4W3_vmcppo_TL'

    trained_model = transfer(epochs=100, small_net_args=mh_net_args, big_net_args=net_args, Ops_args=Ops_args,
            Ham_args=Ham_args, learning_rate=1E-4, n_sample=140000, init_type='rand', threads=70,
            state_size=state_size, input_mh_fn=input_mh_fn, load_state0=False, output_fn=output_fn)

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

        psi = torch.squeeze(trained_model(state_onehots.float())).detach().numpy()
        logphis = psi[:,0]
        thetas = psi[:,1]
        probs = np.exp(logphis*2)/np.sum(np.exp(logphis*2))
        print(np.sum(probs))

        sio.savemat('./data/test_data_HS2dtri_L4W3TL.mat',dict(probs=probs, logphis=logphis, thetas=thetas))

    b_check()
