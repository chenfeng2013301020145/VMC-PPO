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
from algos.core import mlp_cnn, get_paras_number, gradient
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
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), n_sample=100, init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], dimensions='1d', batch_size=3000, clip_ratio=0.1,
          sample_division=5, target_dfs=0.1, save_freq=10, net_args=dict(), threads=4, seed=0,
          input_fn=0, load_state0=True, output_fn='test'):
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

    psi_model = mlp_cnn(state_size=state_size, complex_nn=True, **net_args).to(gpu)
    # model for sampling
    mh_model = mlp_cnn(state_size=state_size, complex_nn=True, **net_args)

    logger.info(psi_model)
    logger.info(get_paras_number(psi_model))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)
    
    if input_fn != 0:
        load_model = torch.load(os.path.join('./results', input_fn))
        psi_model.load_state_dict(load_model)
        if load_state0:
            fn_name = os.path.split(os.path.split(input_fn)[0])
            mat_content = sio.loadmat(os.path.join('./results',fn_name[0], 'state0.mat'))
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
    
    # define the loss function according to the energy functional in GPU
    def compute_loss_energy(model, data):
        state, count, logphi0  = data['state'], data['count'], data['logphi0']
        op_states, op_coeffs = data['update_states'], data['update_coeffs']

        psi = model(state.float())
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)

        # calculate the weights of the energy from important sampling
        delta_logphi = logphi - logphi0[..., None]
        delta_logphi = delta_logphi - delta_logphi.mean()
        count_norm = (count[...,None]/count.sum()).detach()
        weights = count_norm*torch.exp(delta_logphi*2).detach()
        clip_ws = count_norm*torch.clamp(torch.exp(delta_logphi*2), 1-clip_ratio, 1+clip_ratio).detach()
        
        # calculate the coeffs of the energy
        n_sample = op_states.shape[0]
        n_updates = op_states.shape[1]
        op_states = op_states.reshape([-1, Dp]+single_state_shape)
        psi_ops = model(op_states.float())
        logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
        theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

        delta_logphi_os = logphi_ops - logphi*torch.ones_like(logphi_ops)
        delta_theta_os = theta_ops - theta*torch.ones_like(theta_ops)
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

        return torch.stack((loss_re_re, loss_re_im, loss_im_re, loss_im_im), dim=0)

    def regular_backward(data):
        op_model = copy.deepcopy(psi_model)
        params, names = extract_weights(op_model)

        def forward(*new_param):
            load_weights(op_model, names, new_param)
            loss = compute_loss_energy(op_model, data)
            return loss

        dydws = jacobian(forward, params, vectorize=True) 
        cnt = 0

        psi_model.zero_grad()
        for name, p in psi_model.named_parameters():
            if name.split(".")[3] == 'conv_re':
                p.grad = dydws[cnt][0] + dydws[cnt + 2][1]
            elif  name.split(".")[2] == 'linear_re':
                p.grad = dydws[cnt][0] + dydws[cnt + 1][1]
            elif name.split(".")[3] == 'conv_im':
                p.grad = dydws[cnt - 2][2] + dydws[cnt][3]
            elif name.split(".")[2] == 'linear_im':
                p.grad = dydws[cnt - 1][2] + dydws[cnt][3]
            else:
                raise ValueError('Miss update layer: {}'.format(name))
            cnt += 1
        return 

    # setting optimizer in GPU
    optimizer = torch.optim.Adam(psi_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500], gamma=0.1)

    def update(batch_size, target):
        data = buffer.get(batch_size=batch_size)
        # off-policy update
        for i in range(n_optimize):
            #data = buffer.get(batch_size=batch_size)
            optimizer.zero_grad()
            dfs = get_fubini_study_distance(data)
            
            # wn = deltafid
            if dfs > target:
                logger.warning(
                    'early stop at step={} as reaching maximal FS distance'.format(i))
                break

            regular_backward(data)
            optimizer.step()

        return dfs

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('mean_spin: {}'.format(MHsampler._state0_v))
    logger.info('Start training:')
    DFS = 0

    for epoch in range(epochs):
        sample_tic = time.time()
        # sync parameters
        MHsampler._model.load_state_dict(psi_model.state_dict())
        if epoch == 0:
            MHsampler.first_warmup()
        # update the mh_model
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
        DFS = update(IntCount, target_dfs)
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
        # print training informaition
        logger.info('Epoch: {}, AvgE: {:.5f}, StdE: {:.5f}, Lr: {:.2f}, DFS: {:.5f}, IntCount: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
                    format(epoch, AvgE/TolSite, StdE, lr/learning_rate, DFS, IntCount, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))
        
        # save the trained NN parameters
        if epoch % save_freq == 0 or epoch == epochs - 1:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(psi_model.state_dict(), os.path.join(save_dir, 'model_'+str(epoch)+'.pkl'))
            sio.savemat(os.path.join(output_dir, 'state0.mat'), {'state0': MHsampler.single_state0})

    logger.info('Finish training.')

    return psi_model.to(cpu), MHsampler.single_state0, AvgE
