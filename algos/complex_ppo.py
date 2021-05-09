# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version 2.0 updates: compatible with 2d system, replay buffer
# version 3.0 updates: double neural networks (real and imag)
# ppo-clip: early stop with Fubini-Study distance
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from sampler.mcmc_sampler_complex import MCsampler
from algos.core import mlp_cnn, mlp_cnn_sym, get_paras_number, gradient
from utils import SampleBuffer, get_logger, _get_unique_states, extract_weights, load_weights
from torch.autograd.functional import jacobian
import scipy.io as sio
import time
import copy
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
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), n_sample=80, init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], dimensions='1d', batch_size=3000, clip_ratio=0.1,
          sample_division=5, target_dfs=0.01, save_freq=10, sample_freq=5, net_args=dict(), threads=4, 
          seed=0, input_fn=0, load_state0=True, output_fn='test'):
    """
    main training process
    wavefunction: psi = phi*exp(1j*theta)
    output of the CNN network: logphi, theta

    Args:
        epochs (int): Number of epochs of interaction.
        
        Ops_args (dict): setup the names of operators.
        
        Ham_args (dict): setup the Hamiltonian.
        
        init_type (str): set the type of generated initial states

        n_sample (int): Number of sampling in each epoch.

        n_optimize (int): Number of update in each epoch.

        learning_rate: learning rate for Adam.

        state_size: size of a single state, (N, Dp) or (L, W, Dp).

        save_freq: frequency of saving.

        Dp: physical index.

        N or L, W: length of 1d lattice or length, with of 2d lattice
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

    psi_model, name_index = mlp_cnn_sym(state_size=state_size, complex_nn=True, **net_args)
    psi_model.to(gpu)
    # model for sampling
    mh_model, _ = mlp_cnn_sym(state_size=state_size, complex_nn=True, **net_args)

    logger.info(psi_model)
    logger.info(get_paras_number(psi_model))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)
    
    if input_fn != 0:
        load_model = torch.load(os.path.join('../results', input_fn))
        psi_model.load_state_dict(load_model)
        if load_state0:
            fn_name = os.path.split(os.path.split(input_fn)[0])
            mat_content = sio.loadmat(os.path.join('../results',fn_name[0], 'state0.mat'))
            MHsampler.single_state0 = mat_content['state0']

    # mean energy from importance sampling in GPU
    def _energy_ops(sample_division):
        data = buffer.get(batch_type='equal', sample_division=sample_division)
        states, counts  = data['state'], data['count']
        op_states, op_coeffs = data['update_states'], data['update_coeffs']
        
        with torch.no_grad():
            n_sample = op_states.shape[0]
            n_updates = op_states.shape[1]
            op_states = op_states.reshape([-1, Dp] + single_state_shape)

            psi_ops = psi_model(op_states.float())
            logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
            theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

            psi = psi_model(states.float())
            logphi = psi[:, 0].reshape(len(states), -1)
            theta = psi[:, 1].reshape(len(states), -1)

            delta_logphi_os = logphi_ops - logphi*torch.ones(logphi_ops.shape, device=gpu)
            delta_theta_os = theta_ops - theta*torch.ones(theta_ops.shape, device=gpu)
            Es_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)
            Es_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)

            return (Es_real*counts).sum().to(cpu), ((Es_real**2 + Es_imag**2)*counts).sum().to(cpu)
  
    def get_fubini_study_distance(data):
        state, count, logphi0, theta0  = data['state'], data['count'], data['logphi0'], data['theta0']
        with torch.no_grad():
            psi = psi_model(state.float())
            logphi = psi[:, 0].reshape(len(state), -1)
            theta = psi[:, 1].reshape(len(state), -1)
            
            deltalogphi = logphi - logphi0[...,None]
            deltalogphi = deltalogphi - deltalogphi.mean()
            deltatheta = theta - theta0[...,None]
            deltatheta = deltatheta - deltatheta.mean()
            
            phiold_phinew = (count[...,None]*torch.exp(deltalogphi)*torch.exp(1j*deltatheta)).sum()
            phinew_phiold = phiold_phinew.conj()
            phiold_phiold = count.sum()
            phinew_phinew = (count[...,None]*torch.exp(2*deltalogphi)).sum()
            
            dfs = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew))
        return dfs.abs()**2
    
    def target_fubini_study_distance(EGE, AvgE, AvgE2, tau):
        EG = min(-0.5, EGE)*TolSite
        AvgE = AvgE - EG
        AvgE2 = AvgE2 - 2*EG*AvgE + EG**2
        with torch.no_grad():
            phiold_phinew = 1 - tau*AvgE
            phiold_phiold = 1
            phinew_phinew = 1 - 2*tau*AvgE + tau**2*AvgE2
            dfs = np.arccos(np.sqrt(phiold_phinew**2/phiold_phiold/phinew_phinew))
        return dfs**2
    
    # define the loss function according to the energy functional in GPU
    def compute_loss_energy(model, data):
        state, count, logphi0  = data['state'], data['count'], data['logphi0']
        op_states, op_coeffs = data['update_states'], data['update_coeffs']
        # ops_real, ops_imag = data['ops_real'], data['ops_imag']

        psi = model(state)
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)

        # calculate the weights of the energy from important sampling
        delta_logphi = logphi - logphi0[..., None]
        delta_logphi = delta_logphi - delta_logphi.mean()
        weights = count[...,None]*torch.exp(delta_logphi*2)
        weights = (weights/weights.sum()).detach()
        clip_ws = count[...,None]*torch.clamp(torch.exp(delta_logphi*2), 
                                         1-clip_ratio, 1+clip_ratio)
        clip_ws = (clip_ws/clip_ws.sum()).detach()
        
        # calculate the coeffs of the energy
        n_sample = op_states.shape[0]
        n_updates = op_states.shape[1]
        op_states = op_states.reshape([-1, Dp]+single_state_shape)
        psi_ops = model(op_states.float())
        logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
        theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

        delta_logphi_os = logphi_ops - logphi*torch.ones_like(logphi_ops)
        # delta_logphi_os = torch.clamp(delta_logphi_os, -30, np.log(0.25*n_sample))
        delta_theta_os = theta_ops - theta*torch.ones_like(theta_ops)
        # delta_theta_os = torch.fmod(delta_logphi_os, np.pi)
        ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1).detach()
        ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1).detach()
        
        # calculate the mean energy
        me_real = (weights*ops_real[..., None]).sum().detach()
        cme_real = (clip_ws*ops_real[..., None]).sum().detach()
        
        E_re_re = ops_real[..., None]*logphi - me_real*logphi + ops_imag[..., None]*theta
        cE_re_re = ops_real[..., None]*logphi - cme_real*logphi + ops_imag[..., None]*theta
        loss_re_re = 0.5*torch.max(weights*E_re_re, clip_ws*cE_re_re).sum()

        E_re_im = ops_real[..., None]*theta - me_real*theta - ops_imag[..., None]*logphi
        cE_re_im = ops_real[..., None]*theta - cme_real*theta - ops_imag[..., None]*logphi
        loss_re_im = 0.5*torch.max(weights*E_re_im, clip_ws*cE_re_im).sum()

        E_im_re = -ops_real[..., None]*theta + me_real*theta + ops_imag[...,None]*logphi
        cE_im_re = -ops_real[..., None]*theta + cme_real*theta + ops_imag[...,None]*logphi
        loss_im_re = 0.5*torch.max(weights*E_im_re, clip_ws*cE_im_re).sum()

        E_im_im = ops_real[..., None]*logphi - me_real*logphi + ops_imag[..., None]*theta
        cE_im_im = ops_real[..., None]*logphi - cme_real*logphi + ops_imag[..., None]*theta
        loss_im_im = 0.5*torch.max(weights*E_im_im, clip_ws*cE_im_im).sum()

        return torch.stack((loss_re_re, loss_re_im, loss_im_re, loss_im_im), dim=0), me_real

    def regular_backward(data):
        op_model = copy.deepcopy(psi_model)
        params, names = extract_weights(op_model)

        def forward(*new_param):
            load_weights(op_model, names, new_param)
            loss, _ = compute_loss_energy(op_model, data)
            return loss

        dydws = jacobian(forward, params, vectorize=True) 
        cnt = 0

        psi_model.zero_grad()
        for name, p in psi_model.named_parameters():
            if name.split(".")[3 + name_index] == 'conv_re':
                p.grad = dydws[cnt][0] + dydws[cnt + 2][1]
            elif  name.split(".")[2 + name_index] == 'linear_re':
                p.grad = dydws[cnt][0]  + dydws[cnt + 1][1]
            elif name.split(".")[3 + name_index] == 'conv_im':
                p.grad = dydws[cnt - 2][2] + dydws[cnt][3]
            elif name.split(".")[2 + name_index] == 'linear_im':
                p.grad = dydws[cnt][3]  + dydws[cnt - 1][2]
            else:
                raise ValueError('Miss update layer: {}'.format(name))
            cnt += 1
        return 

    # setting optimizer in GPU
    optimizer = torch.optim.Adam(psi_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500], gamma=0.1)

    def update(batch_size, target):
        # full samples for small systems
        data = buffer.get(batch_size=batch_size, get_eops=False)
        # off-policy update
        for i in range(n_optimize):
            # random batch for large systems
            # data = buffer.get(batch_size=batch_size, batch_type='rand', get_eops=True)
            optimizer.zero_grad()
            dfs = get_fubini_study_distance(data)
            _, me = compute_loss_energy(psi_model,data)
            if dfs > target:
                logger.warning(
                    'early stop at step={} as reaching maximal FS distance'.format(i))
                break

            regular_backward(data)
            optimizer.step()

        return dfs, me

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('mean_spin: {}'.format(MHsampler._state0_v))
    logger.info('Start training:')
    MHsampler.basis_warmup_sample = 10*threads
    DFS = 0
    TDFS = target_dfs
    # get initial samples
    # MHsampler._model.load_state_dict(psi_model.state_dict())
    # MHsampler.first_warmup()
    # states, logphis, update_states, update_coeffs = MHsampler.get_new_samples()
    # # using unique states to reduce memory usage.
    # states, _, counts, update_states, update_coeffs = _get_unique_states(states, logphis,
    #                                                         update_states, update_coeffs)
    # n_real_sample = MHsampler._n_sample

    for epoch in range(epochs):
        sample_tic = time.time()
        if epoch > epochs - 100:
            target = min(0.1*target_dfs, TDFS)
            n_optimize = 20
        else:
            target = min(target_dfs, TDFS)
        
        # get new samples from MCMC smapler
        if DFS > 10*target:
            MHsampler._warmup = True
        else:
            MHsampler._warmup = False
        # sync parameters and update the mh_model
        MHsampler._model.load_state_dict(psi_model.state_dict())
        states, logphis, update_states, update_coeffs = MHsampler.get_new_samples()
        n_real_sample = MHsampler._n_sample
        # using unique states to reduce memory usage.
        states, _, counts, update_states, update_coeffs = _get_unique_states(states, logphis,
                                                                update_states, update_coeffs)

        psi = psi_model(torch.from_numpy(states).float().to(gpu))
        logphis = psi[:, 0].reshape(len(states)).cpu().detach().numpy()
        thetas = psi[:, 1].reshape(len(states)).cpu().detach().numpy()
        buffer.update(states, logphis, thetas, counts, update_states, update_coeffs)
        # buffer.get_energy_ops(model=psi_model, Dp=Dp, single_state_shape=single_state_shape)

        IntCount = len(states)
        sample_toc = time.time()

        # ------------------------------------------GPU------------------------------------------
        op_tic = time.time()
        DFS, ME = update(IntCount, target)
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

        # adjust the learning rate
        scheduler.step()
        lr = scheduler.get_last_lr()[-1]
        EGE = AvgE/TolSite - 0 if np.isnan(StdE) else AvgE/TolSite - StdE
        TDFS = target_fubini_study_distance(EGE, AvgE, AvgE2, lr*n_optimize)
        # print training informaition
        logger.info('Epoch: {}, AvgE: {:.5f}, ME: {:.5f}, StdE: {:.5f}, Lr: {:.2f}, DFS: {:.5f}, TDFS: {:.5f}, IntCount: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
                    format(epoch, AvgE/TolSite, ME/TolSite, StdE, lr/learning_rate, DFS, TDFS, IntCount, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))
        
        # save the trained NN parameters
        if epoch % save_freq == 0 or epoch == epochs - 1:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(psi_model.state_dict(), os.path.join(save_dir, 'model_'+str(epoch)+'.pkl'))
            sio.savemat(os.path.join(output_dir, 'state0.mat'), {'state0': MHsampler.single_state0})

    logger.info('Finish training.')

    return psi_model.to(cpu), MHsampler.single_state0, AvgE
