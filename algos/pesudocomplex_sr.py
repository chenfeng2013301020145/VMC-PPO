# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version 2.0 updates: compatible with 2d system, replay buffer
# version 3.0: double CNN (real and imag), imaginary time propagation (Stochastic Reconfiguration, ISGO)
# evaluate Oks, Skk_matrix with torch.autograd.functional.jacobian
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from sampler.mcmc_sampler_complex import MCsampler
from algos.core import mlp_cnn, get_paras_number
from utils import SampleBuffer, get_logger, _get_unique_states, extract_weights, load_weights
from torch.autograd.functional import jacobian
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

# ------------------------------------------------------------------------
# main training function
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), n_sample=100, init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], resample_condition=50, dimensions='1d', batch_size=1000,
          sample_division=5, target_dfs=0.1, save_freq=10, net_args=dict(), threads=4, epsilon=0.1,
          input_fn=0, seed=0, load_state0=True, output_fn='test'):
    """
    main training process
    wavefunction: psi = phi*exp(1j*theta)
    output of the CNN network: logphi, theta

    Args:
        epochs (int): Number of epochs of interaction.

        n_sample (int): Number of sampling in each epoch.

        n_optimize (int): Number of update in each epoch.

        lr: learning rate for Adam.

        state_size: size of a single state, [n_sites, Dp].

        save_freq: frequency of saving.

        Dp: physical index.

        N or L, W: length of 1d lattice or length and with of 2d lattice
    """
    seed += 1000*np.sum(np.arange(threads))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = os.path.join('../results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')
    logger = get_logger(os.path.join(output_dir, 'exp_log.txt'))

    if dimensions == '1d':
        TolSite = state_size[0]  # number of sites
        single_state_shape = [state_size[0]]
    else:
        TolSite = state_size[0]*state_size[1]
        single_state_shape = [state_size[0], state_size[1]]
    Dp = state_size[-1]  # number of physical spins

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator
    buffer = SampleBuffer(gpu) 

    psi_model = mlp_cnn(state_size=state_size, output_size=2, complex_nn=False,
                    **net_args).to(gpu)

    mh_model = mlp_cnn(state_size=state_size, output_size=2, complex_nn=False,
                    **net_args)
    logger.info(psi_model)
    logger.info(get_paras_number(psi_model))
    logger.info('epsilion: {}'.format(epsilon))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)

    if input_fn != 0:
        load_model = torch.load(os.path.join('./results', input_fn))
        psi_model.load_state_dict(load_model)
        # theta_model.load_state_dict(load_models['theta_model']) 
        if load_state0:
            fn_name = os.path.split(os.path.split(input_fn)[0])
            mat_content = sio.loadmat(os.path.join('./results',fn_name[0], 'state0.mat'))
            MHsampler.single_state0 = mat_content['state0']

    # mean energy from importance sampling in GPU
    def _energy_ops(sample_division):
        data = buffer.get(batch_type='equal', sample_division=sample_division)
        states, counts, op_states, op_coeffs = data['state'], data['count'], data['update_states'], data['update_coeffs']

        with torch.no_grad():
            n_sample = op_states.shape[0]
            n_updates = op_states.shape[1]
            op_states = op_states.reshape([-1, Dp] + single_state_shape)

            psi_ops = psi_model(op_states)
            logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
            theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

            psi = psi_model(states)
            logphi = psi[:, 0].reshape(len(states), -1)
            theta = psi[:, 1].reshape(len(states), -1)

            delta_logphi_os = logphi_ops - logphi*torch.ones_like(logphi_ops)
            delta_theta_os = theta_ops - theta*torch.ones_like(theta_ops)
            Es = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)

            return (Es*counts).sum().to(cpu), ((Es**2)*counts).sum().to(cpu)
        
    def get_fubini_study_distance(data):
        state, count, logphi0, theta0  = data['state'], data['count'], data['logphi0'], data['theta0']
        with torch.no_grad():
            psi = psi_model(state.float())
            logphi = psi[:, 0].reshape(len(state), -1)
            theta = psi[:, 1].reshape(len(state), -1)
            
            deltalogphi = logphi - logphi0[...,None]
            deltatheta = theta - theta0[...,None]
            
            phiold_phinew = (count[...,None]*torch.exp(deltalogphi)*torch.exp(1j*deltatheta)).sum()
            phinew_phiold = phiold_phinew.conj()
            phiold_phiold = count.sum()
            phinew_phinew = (count[...,None]*torch.exp(2*deltalogphi)).sum()
            
            fsd = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew))
        return fsd.abs()

    # setting optimizer in GPU
    # optimizer = torch.optim.Adam(logphi_model.parameters(), lr=learning_rate)
    # imaginary time propagation: delta_t = learning_rate
    def update_one_step(data, epsilon):
        state, count, logphi0  = data['state'], data['count'], data['logphi0']
        op_states, op_coeffs = data['update_states'], data['update_coeffs']
        psi = psi_model(state)
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)

        # calculate the weights of the energy from important sampling
        delta_logphi = logphi - logphi0[..., None]

        # delta_logphi = delta_logphi - delta_logphi.mean()*torch.ones(delta_logphi.shape)
        delta_logphi = delta_logphi - delta_logphi.mean()
        weights = count[..., None]*torch.exp(delta_logphi * 2)
        weights_norm = weights.sum()
        weights = (weights/weights_norm).detach()
        
        n_sample = op_states.shape[0]
        n_updates = op_states.shape[1]
        op_states = op_states.reshape([-1, Dp] + single_state_shape)
        psi_ops = psi_model(op_states)
        logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
        theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

        delta_logphi_os = logphi_ops - logphi*torch.ones_like(logphi_ops)
        # delta_logphi_os = torch.clamp(delta_logphi_os, max=5)
        delta_theta_os = theta_ops - theta*torch.ones_like(theta_ops)
        ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)
        ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)
        
        with torch.no_grad():
            ops = ops_real + 1j*ops_imag # batch_size
            mean_e = (ops_real[...,None]*weights).sum(0)
        # update parameters with gradient descent
        # copy the model parameters
        op_psi_model = copy.deepcopy(psi_model)
        params, names = extract_weights(op_psi_model)

        def forward(*new_param):
            load_weights(op_psi_model, names, new_param)
            out = op_psi_model(state)
            return out

        dydws = jacobian(forward, params, vectorize=True) # a tuple contain all grads
        cnt = 0
        tic = time.time()
        psi_model.zero_grad()
        for param in psi_model.parameters():
            param_len = len(param.data.reshape(-1))
            dydws_layer = dydws[cnt].reshape(n_sample,2,-1)
            with torch.no_grad():
                grads_real = dydws_layer[:,0,:] # jacobian of d logphi wrt d w
                grads_imag = dydws_layer[:,1,:] # jacobian of d theta wrt d w
                Oks = grads_real + 1j*grads_imag
                Oks_conj = grads_real - 1j*grads_imag
                OO_matrix = Oks_conj.reshape(n_sample, 1, param_len)*Oks.reshape(n_sample, param_len, 1)
            
            Oks = Oks*weights
            Oks_conj = Oks_conj*weights
            Skk_matrix = (OO_matrix*weights[...,None]).sum(0) - Oks_conj.sum(0)[..., None]*Oks.sum(0)
            Skk_matrix = 0.5*(Skk_matrix + Skk_matrix.t().conj()) + epsilon*torch.eye(Skk_matrix.shape[0], device=gpu)
            # calculate Fk
            Fk = (ops[...,None]*Oks_conj).sum(0) - mean_e*(Oks_conj).sum(0)
            # update_k = torch.linalg.solve(Skk_matrix, Fk)
            update_k, _ = torch.solve(Fk[...,None], Skk_matrix)
            param.grad = update_k.real.reshape(param.data.shape)
            cnt += 1
        t = time.time() - tic
        return weights_norm/count.sum(), t

    optimizer = torch.optim.Adam(psi_model.parameters(), lr=learning_rate)
    
    def update(IntCount, epsilon, epoch):
        data = buffer.get(batch_size=IntCount)
        t = 0
        for _ in range(n_optimize):
            optimizer.zero_grad()
            _, dt = update_one_step(data, epsilon)
            dfs = get_fubini_study_distance(data)
            optimizer.step()
            t += dt            
        # logger.info('jacobian_time: {}'.format(t))
        return dfs

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('mean_spin: {}'.format(MHsampler._state0_v/threads))
    logger.info('Start training:')
    warmup_n_sample = n_sample // 10
    DFS = 0

    for epoch in range(epochs):
        sample_tic = time.time()
        MHsampler._n_sample = warmup_n_sample
        # print(logphi_model.state_dict())
        MHsampler._model.load_state_dict(psi_model.state_dict())
        if epoch == 0:
            MHsampler.first_warmup()
        if DFS > 2*target_dfs:
            MHsampler._warmup = True
        else:
            MHsampler._warmup = False
            
        states, logphis, update_states, update_coeffs = MHsampler.parallel_mh_sampler()
        n_real_sample = MHsampler._n_sample

        # using unique states to reduce memory usage.
        states, _, counts, update_states, update_coeffs = _get_unique_states(states, logphis,
                                                                            update_states, update_coeffs)
        psi = psi_model(torch.from_numpy(states).float().to(gpu))
        logphis = psi[:, 0].reshape(len(states)).cpu().detach().numpy()
        thetas = psi[:, 1].reshape(len(states)).cpu().detach().numpy()
        buffer.update(states, logphis, thetas, counts, update_states, update_coeffs)

        IntCount = len(states)

        sample_toc = time.time()

        # ------------------------------------------GPU------------------------------------------
        op_tic = time.time()
        DFS = update(IntCount, epsilon, epoch)
        op_toc = time.time()
        
        sd = 1 if IntCount < batch_size else sample_division
        avgE = torch.zeros(sd)
        avgE2 = torch.zeros(sd)
        for i in range(sd):    
            avgE[i], avgE2[i] = _energy_ops(sd)
        # ---------------------------------------------------------------------------------------
        # average over all samples
        AvgE = avgE.sum().numpy()/n_real_sample
        AvgE2 = avgE2.sum().numpy()/n_real_sample
        StdE = np.sqrt(AvgE2 - AvgE**2)/TolSite

        # print training informaition
        logger.info('Epoch: {}, AvgE: {:.5f}, StdE: {:.5f}, Lr: {:.5f}, Ep: {:.5f}, DFS: {:.3f}, IntCount: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
                    format(epoch, AvgE/TolSite, StdE, learning_rate, epsilon, DFS, IntCount, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))

        epsilon *= 0.998
        # learning_rate *= 0.99

        # save the trained NN parameters
        if epoch % save_freq == 0 or epoch == epochs - 1:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(psi_model.state_dict(), os.path.join(save_dir, 'model_'+str(epoch)+'.pkl'))
            sio.savemat(os.path.join(output_dir, 'state0.mat'), {'state0': MHsampler.single_state0})

        if warmup_n_sample != n_sample:
            # first 5 epochs are used to warm up due to the limitations of memory
            warmup_n_sample += n_sample // 10

    logger.info('Finish training.')

    return psi_model.to(cpu), MHsampler.single_state0, AvgE
