# encoding: utf-8
import sys
sys.path.append('..')

import torch.nn as nn
import torch
import numpy as np
from utils_ppo import rot60
import copy
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
# from snake.activations import Snake
# from utils import extract_weights,load_weights

def periodic_padding(x, kernel_size, dimensions, mode='circular'):
    if dimensions == '1d':
        # shape of real x: (batch_size, Dp, N) and complex x: (batch_size, 2, Dp, N)
        if len(x.shape) == 3:
            pad = (0, kernel_size-1)
        else:
            pad = (0, kernel_size-1, 0, 0)
    else:
        # shape of real x: (batch_size, Dp, Length, Width)
        #      and complex x: (batch_size, 2, Dp, Length, Width)
        if len(x.shape) == 4:
            pad = (0, kernel_size[1]-1, 0, kernel_size[0] - 1)
        else:
            pad = (0, kernel_size[1]-1, 0, kernel_size[0] - 1, 0, 0)
    return nn.functional.pad(x, pad, mode)

def get_paras_number(net, filter_num=0):
    # total_num = sum(p.numel() for p in net.parameters())
    # trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num = 0
    trainable_num = 0
    for name,p in net.named_parameters():
        if 'conv' in name and 'weight' in name and filter_num>0:
            total_num += p.shape[0]*p.shape[1]*p.shape[2]*(filter_num)
        else:
            total_num += p.numel()

        if p.requires_grad:
            if 'conv' in name and 'weight' in name and filter_num>0:
                trainable_num += p.shape[0]*p.shape[1]*p.shape[2]*(filter_num)
            else:
                trainable_num += p.numel()
    if filter_num == 21:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'custom_sq'}
    elif filter_num == 19:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'custom_tri'}
    else:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'default'}

def gradient(y, x, grad_outputs=None):
    """compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def complex_init(model, custom_filter):
    # cnt = -1
    for name, p in model.named_parameters():
        #print(name, p.data.shape)
        # if int(name.split(".")[0]) - cnt == 1:
        #     rho = torch.from_numpy(np.random.rayleigh(1, size=p.data.shape)).float()
        #     theta = np.random.uniform(-np.pi, +np.pi, size=p.data.shape)
        # cnt += 1
        if 'weight' in name:
            #w = np.sqrt((p.data.shape[-1] + p.data.shape[-2]))
            if 'conv_re' in name: 
                p.data = p.data*custom_filter
        # elif 'bias' in name:
        #     nn.init.zeros_(p.data)

    return

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse_indices = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    #unique, inverse_indices = torch.unique_consecutive(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype,
                        device=inverse_indices.device)
    inverse, perm = inverse_indices.flip([0]), perm.flip([0])
    indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, indices, inverse_indices

# COMPLEX NEURAL NETWORK
# ----------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imag = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        return torch.stack((real, imag),dim=1)

class RealLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(RealLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0])
        imag = self.linear_im(x[:,1])
        output = torch.stack((real, imag),dim=1)
        return output

class ComplexLnCosh(nn.Module):
    def __init__(self):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexLnCosh, self).__init__()

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        z = torch.log(2*torch.cosh(z))
        return torch.stack((z.real, z.imag), dim=1)

class ComplexTanh(nn.Module):
    def __init__(self):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexTanh, self).__init__()

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        z = torch.tanh(z)
        return torch.stack((z.real, z.imag), dim=1)

class ComplexReLU(nn.Module):
    def __init__(self, relu_type='zReLU', alpha=0.35):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexReLU, self).__init__()
        self.type = relu_type
        self.act_re = nn.Parameter(torch.tensor(0.35))
        self.act_re.requires_grad = False
        self.act_im = nn.Parameter(torch.tensor(alpha))
        self.act_im.requires_grad = True

    def forward(self, x, *args):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        if self.type == 'softplus':
            return torch.stack((nn.functional.softplus(real), nn.functional.softplus(imag)), dim=1)
        elif self.type == 'selu':
            return torch.stack((nn.functional.selu(real), nn.functional.selu(imag)), dim=1)
        elif self.type == 'serelu':
            return torch.stack((nn.functional.selu(real), nn.functional.relu(imag)), dim=1)
        elif self.type == 'snake':
            # real = nn.functional.selu(real)
            real = real + (1 - torch.cos(2*self.act_re*real))/(2*self.act_re)
            #torch.sin(self.act_re*real)*torch.sin(self.act_re*real)/self.act_re
            imag = imag + (1 - torch.cos(2*self.act_im*imag))/(2*self.act_im) 
            #torch.sin(self.act_im*imag)*torch.sin(self.act_im*imag)/self.act_im
            # imag = nn.functional.relu(imag + torch.sin(self.act*imag)*torch.sin(self.act*imag)/self.act)
            # imag = imag + (1 - torch.cos(2*self.act*imag))/(2*self.act)
            return torch.stack((real, imag), dim=1)
        else:
            return torch.stack((torch.relu(real), torch.relu(imag)), dim=1)

class Snake(nn.Module):
    def __init__(self, alpha=0.35, require_grad=False):
        super(Snake, self).__init__()
        #self.act = nn.Parameter(torch.tensor(alpha))
        #self.act.requaires_grad = require_grad
        self.act = alpha

    def forward(self, x, *args):
        return x + (1 - torch.cos(2*self.act*x))/(2*self.act)

# COMPLEX CNN
#--------------------------------------------------------------------
def translation_phase(z, k, dimensions):
    xdevice = z.device
    n_sample = z.shape[0]
    F = z.shape[1]
    real = z.real
    imag = z.imag
    if dimensions == '1d':
        N = z.shape[-1]
        vec = torch.exp(1j*2*np.pi*k[0]*torch.arange(N, device=xdevice)/N)
        vec_real = vec.real.repeat(n_sample, F, 1)
        vec_imag = vec.imag.repeat(n_sample, F, 1)
        real_part = real*vec_real - imag*vec_imag
        imag_part = real*vec_imag + imag*vec_real
    else:
        L = z.shape[-2]
        W = z.shape[-1]
        mat = (torch.exp(1j*2*np.pi*k[1]*torch.arange(W, device=xdevice)/W)
            *torch.exp(1j*2*np.pi*k[0]*torch.arange(L, device=xdevice)/L)[...,None])
        mat_real = mat.real.repeat(n_sample, F, 1, 1)
        mat_imag = mat.imag.repeat(n_sample, F, 1, 1)
        real_part = real*mat_real - imag*mat_imag
        imag_part = real*mat_imag + imag*mat_real
    return real_part + 1j*imag_part

class CNN_real_layer(nn.Module):
    def __init__(self,K,F_in,F_out,stride,layer_name='mid',
                 pbc=True, dimensions='1d', custom_filter=False, 
                 lattice_shape=None, device='cpu'):
        super(CNN_real_layer,self).__init__()
        self.K = [K,K] if type(K) is int and dimensions=='2d' else K
        self._pbc = pbc
        self.layer_name = layer_name
        self.dimensions = dimensions
        self.stride = [stride, stride] if type(stride) is int and dimensions=='2d' else stride

        if dimensions == '1d':
            self.conv = nn.Conv1d(F_in,F_out,K,stride,0)
            #self.conv
        else:
            if self.layer_name == '1st':
                self.conv = P4MConvZ2(in_channels=F_in,
                                    out_channels=F_out,
                                    kernel_size=K,
                                    stride=stride,
                                    padding=0)
            else:
                self.conv = P4MConvP4M(in_channels=F_in,
                                    out_channels=F_out,
                                    kernel_size=K,
                                    stride=stride,
                                    padding=0)

        self.custom_filter = custom_filter

        if lattice_shape == 'sq':
            """
            tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    ])
            """
            self.filter = torch.tensor([
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    ], device=device).reshape(1,1,5,5)
        else:
            """
            tensor([
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    ])
            """
            self.filter = torch.tensor([
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    ], device=device).reshape(1,1,1,5,5)

    def forward(self, x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions=self.dimensions, mode='circular')
        else:
            x = periodic_padding(x, self.K, dimensions=self.dimensions, mode='constant')

        if self.custom_filter:
            self.conv.weight.data = self.conv.weight.data*self.filter

        x = self.conv(x)
        return x

class OutPut_real_layer(nn.Module):
    def __init__(self,F,dimensions='1d',output_mode='phi',inverse_symmetry=False, Dp=2):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut_real_layer,self).__init__()
        #self.K = [K,K] if type(K) is int and dimensions=='2d' else K
        self.F = F
        #self._pbc=pbc
        self.dimensions = dimensions
        self.output_mode = output_mode
        self._inverse_symmetry = inverse_symmetry
        self.Dp = Dp
        #if self.dimensions=='1d':
        #if self.output_mode == 'phi':
        #self.linear = nn.Linear(F, 1, bias=False)

    def forward(self,x,*args):
        # shape of complex x: (batch_size, F, G, L, W)
        # variance scaling
        if self._inverse_symmetry:
            x = x.reshape(-1, self.Dp*x.shape[1], x.shape[-3], x.shape[-2], x.shape[-1])
        norm = np.sqrt(np.prod(x.shape[1:]))

        # logsumexp
        if self.dimensions=='1d':
            return torch.sum(x, dim=[1,2])/norm
        else:
            # return self.linear(torch.sum(x, dim=[2,3,4])/norm).squeeze(-1)
            # if self.output_mode == 'phi':
            #     return torch.sum(x, dim=[1,2,3,4])/norm
            # else:
            #     return torch.sum(x, dim=[1,2,3,4])/norm
            
            # shape of complex x: (batch_size, F, G, L, W)
            if self.output_mode == 'phi':
                #return self.linear(torch.sum(x, dim=[2,3,4])/norm).squeeze(-1)
                return torch.logsumexp(x, dim=[1,2,3,4])/norm
                #return torch.logsumexp(x, dim=[1,2,3,4])/norm
            else:
                #return self.linear(torch.sum(x, dim=[2,3,4])/norm).squeeze(-1)
                return ((torch.exp(1j*x)/norm).sum(dim=[1,2,3,4])).angle()
                #return (torch.exp(1j*x).sum(dim=[1,2,3,4])).angle()
        #theta = torch.sum(z.imag, dim=[1,2,3]) if self.dimensions=='1d' else torch.sum(z.imag, dim=[1,2,3,4])
        #return torch.stack((logphi, theta), dim=1)   


# class CNN_complex_layer(nn.Module):
#     def __init__(self,Kre,Kim,F_in,F_out,stride,layer_name='mid',
#                  pbc=True, dimensions='1d', custom_filter=False, 
#                  lattice_shape=None, device='cpu'):
#         """
#         Dp = 1: value encoding
#         Dp > 1: onehot encoding
#         """
#         super(CNN_complex_layer,self).__init__()
#         self.Kre = [Kre,Kre] if type(Kre) is int and dimensions=='2d' else Kre
#         self.Kim = [Kim,Kim] if type(Kim) is int and dimensions=='2d' else Kim
#         self._pbc = pbc
#         self.layer_name = layer_name
#         self.dimensions = dimensions
#         self.stride = [stride, stride] if type(stride) is int and dimensions=='2d' else stride

#         if self.layer_name == '1st':
#             self.conv_re = P4MConvZ2(in_channels=F_in,
#                                      out_channels=F_out,
#                                      kernel_size=Kre,
#                                      stride=stride,
#                                      padding=0)

#             self.conv_im = P4MConvZ2(in_channels=F_in,
#                                      out_channels=F_out,
#                                      kernel_size=Kim,
#                                      stride=stride,
#                                      padding=0)
#         else:
#             self.conv_re = P4MConvP4M(in_channels=F_in,
#                                      out_channels=F_out,
#                                      kernel_size=Kre,
#                                      stride=stride,
#                                      padding=0)

#             self.conv_im = P4MConvP4M(in_channels=F_in,
#                                      out_channels=F_out,
#                                      kernel_size=Kim,
#                                      stride=stride,
#                                      padding=0)

#         # self.complex_relu = complex_relu
#         self.custom_filter = custom_filter

#         if lattice_shape == 'sq':
#             """
#             tensor([
#                     [0, 0, 1, 0, 0],
#                     [0, 1, 1, 1, 0],
#                     [1, 1, 1, 1, 1],
#                     [0, 1, 1, 1, 0],
#                     [0, 0, 1, 0, 0],
#                     ])
#             """
#             self.filter = torch.tensor([
#                                     [0, 1, 1, 1, 0],
#                                     [1, 1, 1, 1, 1],
#                                     [1, 1, 1, 1, 1],
#                                     [1, 1, 1, 1, 1],
#                                     [0, 1, 1, 1, 0],
#                                     ], device=device).reshape(1,1,1,5,5)
#         else:
#             """
#             tensor([
#                     [1, 1, 1, 0, 0],
#                     [1, 1, 1, 1, 0],
#                     [1, 1, 1, 1, 1],
#                     [0, 1, 1, 1, 1],
#                     [0, 0, 1, 1, 1],
#                     ])
#             """
#             self.filter = torch.tensor([
#                                     [1, 1, 1, 0, 0],
#                                     [1, 1, 1, 1, 0],
#                                     [1, 1, 1, 1, 1],
#                                     [0, 1, 1, 1, 1],
#                                     [0, 0, 1, 1, 1],
#                                     ], device=device).reshape(1,1,1,5,5)

#     def forward(self, x, _only_theta=False, _only_phi=False):
#         if self.layer_name == '1st':
#             xre = x.clone()
#             xim = x.clone()
#         else:
#             xre = x[:,0]
#             xim = x[:,1]

#         if self._pbc:
#             xre = periodic_padding(xre, self.Kre, dimensions=self.dimensions, mode='circular')
#             xim = periodic_padding(xim, self.Kim, dimensions=self.dimensions, mode='circular')
#         else:
#             xre = periodic_padding(xre, self.Kre, dimensions=self.dimensions, mode='constant')
#             xim = periodic_padding(xim, self.Kim, dimensions=self.dimensions, mode='constant')

#         if self.custom_filter:
#             self.conv_re.weight.data = self.conv_re.weight.data*self.filter
#             self.conv_im.weight.data = self.conv_im.weight.data*self.filter
        
#         if _only_phi:
#             xre = self.conv_re(xre)
#             xim = torch.zeros_like(xre)
#         else:
#             xim = self.conv_im(xim)
#             if _only_theta:
#                 xre = torch.zeros_like(xim)
#             else:
#                 xre = self.conv_re(xre)

#         return torch.stack((xre, xim), dim=1)

# class OutPut_complex_layer(nn.Module):
#     def __init__(self,K,F,pbc=True,dimensions='1d',momentum=[0,0]):
#         """
#         output size = 1: logphi
#         output size = 2: logphi, theta
#         """
#         super(OutPut_complex_layer,self).__init__()
#         self.K = [K,K] if type(K) is int and dimensions=='2d' else K
#         self.F = F
#         self._pbc=pbc
#         self.dimensions = dimensions
#         self.momentum = momentum
#         #self.linear = RealLinear(F, 1, bias=False)

#     def forward(self,x,*args):
#         # shape of complex x: (batch_size, 2, F, G, L, W)
#         # variance scaling
#         norm = np.sqrt(np.prod(x.shape[2:]))
#         z = (x[:,0] + 1j*x[:,1])

#         if self._pbc and np.sum(self.momentum) != 0:
#             z = translation_phase(z, k=self.momentum, dimensions=self.dimensions)

#         # z = z.sum(dim=[2,3]) if self.dimensions=='1d' else z.sum(dim=[2,3,4])
#         # #return torch.stack((z.real, z.imag), dim=1)
#         # x = torch.stack((z.real, z.imag), dim=1)
#         # return self.linear(x).squeeze(dim=-1)

#         # logsumexp
#         if self.dimensions=='1d':
#             logphi = torch.logsumexp(z.real, dim=[1,2,3])/norm
#             # theta = (torch.exp(1j*z.imag).sum(dim=[1,2,3])).angle()
#             theta = torch.sum(z.imag, dim=[1,2,3])
#         else:
#             logphi = torch.logsumexp(z.real, dim=[1,2,3,4])/norm
#             # theta = torch.sum(z.imag, dim=[1,2,3,4])/norm
#             # theta = np.pi*torch.sin(torch.sum(z.imag, dim=[1,2,3,4])/norm)
#             # theta = torch.asin(torch.mean(torch.sin(z.imag), dim=[1,2,3,4]))
#             theta = (torch.exp(1j*z.imag).sum(dim=[1,2,3,4])).angle()
#         #theta = torch.sum(z.imag, dim=[1,2,3]) if self.dimensions=='1d' else torch.sum(z.imag, dim=[1,2,3,4])
#         return torch.stack((logphi, theta), dim=1)
    
#--------------------------------------------------------------------
class mySequential(nn.Sequential):
    def forward(self, input, _only_theta=False, _only_phi=False):
        for module in self._modules.values():
            input = module(input, _only_theta, _only_phi)
        return input

# def mlp_cnn(state_size, K, F=[4,3,2], stride0=[1], stride=[1], 
#     relu_type='selu', pbc=True, bias=True, momentum=[1,0], groups=1, alpha=0.35,
#     custom_filter=False, lattice_shape=None, device='cpu'):
#     '''Input: State (batch_size, Dp, N) for 1d lattice,
#                     (batch_size, Dp, L, W) for 2d lattice.
#        Output: [logphis, thetas].
#     '''
#     K = K[0] if type(K) is list and len(K) == 1 else K
#     stride0 = stride0[0] if type(stride0) is list and len(stride0) == 1 else stride0
#     stride = stride[0] if type(stride) is list and len(stride) == 1 else stride
#     dim = len(state_size) - 1
#     dimensions = '1d' if dim == 1 else '2d'
#     # act_size = [state_size[0]] if dim == 1 else [state_size[0], state_size[1]]
#     layers = len(F)

#     Dp = state_size[-1]
#     complex_relu = ComplexReLU(relu_type, alpha=alpha)
#     input_layer = CNN_complex_layer(Kre=K, Kim=K, F_in=Dp, F_out=F[0], stride=stride0, 
#                                 layer_name='1st', dimensions=dimensions, pbc=pbc,
#                                 custom_filter=custom_filter, lattice_shape=lattice_shape, 
#                                 device=device)
#     output_layer = OutPut_complex_layer(K,F[-1], pbc=pbc, dimensions=dimensions, momentum=momentum)

#     # input layer
#     cnn_layers = [input_layer]
#     for i in range(1,layers):
#         cnn_layers += [complex_relu]
#         cnn_layers += [CNN_complex_layer(Kre=K, Kim=K, F_in=F[i-1], F_out=F[i], 
#                                     stride=stride, dimensions=dimensions, pbc=pbc,
#                                     custom_filter=custom_filter, lattice_shape=lattice_shape, 
#                                     device=device)]

#     cnn_layers += [output_layer]

#     return mySequential(*cnn_layers)

def mlp_cnn_single(state_size, K, F=[4,3,2], stride=[1], pbc=True,
    custom_filter=False, lattice_shape=None, device='cpu',output_mode='phi', inverse_symmetry=False):
    '''Input: State (batch_size, Dp, N) for 1d lattice,
                    (batch_size, Dp, L, W) for 2d lattice.
       Output: [logphis, thetas].
    '''
    K = K[0] if type(K) is list and len(K) == 1 else K
    # stride0 = stride0[0] if type(stride0) is list and len(stride0) == 1 else stride0
    stride = stride[0] if type(stride) is list and len(stride) == 1 else stride
    dim = len(state_size) - 1
    dimensions = '1d' if dim == 1 else '2d'
    # act_size = [state_size[0]] if dim == 1 else [state_size[0], state_size[1]]
    layers = len(F)

    Dp = 1 if inverse_symmetry else state_size[-1]
    #act = Snake(alpha=alpha, require_grad=act_require_grad)
    act = nn.SELU()
    input_layer = CNN_real_layer(K=K, F_in=Dp, F_out=F[0], stride=stride, 
                                layer_name='1st', dimensions=dimensions, pbc=pbc,
                                custom_filter=custom_filter, lattice_shape=lattice_shape, 
                                device=device)
    output_layer = OutPut_real_layer(F[-1], dimensions=dimensions, output_mode=output_mode,
                                    inverse_symmetry=inverse_symmetry, Dp=state_size[-1])

    # input layer
    cnn_layers = [input_layer]
    for i in range(1,layers):
        cnn_layers += [act]
        cnn_layers += [CNN_real_layer(K=K, F_in=F[i-1], F_out=F[i], 
                                stride=stride, dimensions=dimensions, pbc=pbc,
                                custom_filter=custom_filter, lattice_shape=lattice_shape, 
                                device=device)]

    cnn_layers += [output_layer]

    return nn.Sequential(*cnn_layers)
# physical symmetry
# -------------------------------------------------------------------------------------------------
# symmetry functions
def translation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    # output shape of pad_x: (batch_size, Dp, 2*N-1) or (batch_size, Dp, 2*L-1, 2*W-1)
    dimension = len(x.shape) - 2
    if dimension == 1:
        N = x.shape[-1]
        pad_x = periodic_padding(x, N, dimensions='1d', mode='circular')
        return pad_x, 1
    else:
        L, W = x.shape[-2], x.shape[-1]
        pad_x = periodic_padding(x, [L, W], dimensions='2d', mode='circular')
        return pad_x, 1

def inverse(x, dim=1):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    N = 2
    x_inverse = torch.flip(x, dims=[dim])
    full_x = torch.stack((x, x_inverse), dim=1).reshape([-1] + list(x.shape[1:]))
    return full_x, N

def identity(x):
    return x, 1

def transpose(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 2
        xT = torch.transpose(x, 2, 3)
        full_x = torch.stack((x, xT), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def reflection(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 2
        xR = torch.flip(x, dims=[2])
        full_x = torch.stack((x, xR), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c6rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 6
        x60 = rot60(x, num=1, dims=[2,3], center=[0])
        x120 = rot60(x, num=2, dims=[2,3], center=[0])
        x180 = rot60(x, num=3, dims=[2,3], center=[0])
        x240 = rot60(x, num=4, dims=[2,3], center=[0])
        x300 = rot60(x, num=5, dims=[2,3], center=[0])
        full_x = torch.stack((x,x60,x120,x180,x240,x300), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c4rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        return c2rotation(x)
    else:
        N = 4
        x90 = torch.rot90(x, 1, dims=[2,3])
        x180 = torch.rot90(x, 2, dims=[2,3])
        x270 = torch.rot90(x, 3, dims=[2,3])
        full_x = torch.stack((x, x90, x180, x270), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c3rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 3
        x120 = rot60(x, num=2, dims=[2,3], center=[1])
        x240 = rot60(x, num=4, dims=[2,3], center=[1])
        full_x = torch.stack((x, x120, x240), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c2rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    N = 2
    if dimension == 1:
        x180 = torch.rot90(x, 2, dims=[2])
    else:
        x180 = torch.rot90(x, 2, dims=[2, 3])
        full_x = torch.stack((x, x180), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

# """
# need to update for larger system
# """
# def get_max_label1d(full_x, N):
#     # input shape of full_x: (batch_size, sym_N, Dp, 2*N-1)
#     full_xv = full_x[:,:,0,:]
#     batch_size, pad_N = full_xv.shape[0], full_xv.shape[-1]
#     pow_list = torch.arange(N-1, -1, -1, device=full_x.device)
#     filters = torch.pow(torch.tensor(1.5, dtype=torch.float64), pow_list).reshape(1,1,N)
#     full_xv = nn.functional.conv1d(full_xv.reshape(-1, 1, pad_N).double(), filters)
#     # max_label = full_xv.reshape(batch_size, -1).argmax(dim=1)
#     max_num, max_label = torch.max(full_xv.reshape(batch_size, -1), dim=1)
#     x = torch.div(max_label, N, rounding_mode='floor')
#     y = max_label%N
#     return [x, y], max_num

# def get_max_label2d(full_x, L, W):
#     # input shape of full_x: (batch_size, sym_N, Dp, 2*L-1, 2*W-1)
#     """
#     Future: act the symmetry function on the filters
#     """
#     full_xv = full_x[:,:,0,:,:]
#     batch_size, pad_L, pad_W = full_xv.shape[0],full_xv.shape[-2], full_xv.shape[-1]
#     pow_list = torch.arange(L*W-1, -1, -1, device=full_x.device)
#     filters = torch.pow(torch.tensor(1.5, dtype=torch.float64), pow_list).reshape(1,1,L,W)
#     full_xv = nn.functional.conv2d(full_xv.reshape(-1, 1, pad_L, pad_W).double(), filters)
#     max_num, max_label = torch.min(full_xv.reshape(batch_size, -1), dim=1)
#     x = torch.div(max_label, (L*W), rounding_mode='floor')
#     y = torch.div(max_label%(L*W), W, rounding_mode='floor')
#     z = max_label%W
#     return [x, y, z], max_num

# -------------------------------------------------------------------------------------------------
# symmetry network
# Group CNN 
class sym_model(nn.Module):
    def __init__(self, state_size, K, F=[4,3,2], stride=[1], device="cpu",
                pbc=True, sym_funcs=[identity], apply_unique=False, alpha=0.35, 
                custom_filter=False, lattice_shape=None, inverse_symmetry=False, precision=torch.float32):
        super(sym_model,self).__init__()

        self.sym_funcs = sym_funcs
        # calculate number of groups 
        net_state_size = state_size[:]
        x0 = torch.zeros(list(np.roll(net_state_size, shift=1)))[None, ...]
        dimensions = len(state_size) - 1
        _, self.sym_N = self.symmetry(x0)
        self.apply_unique = apply_unique
        #self.MPphase = MPphase
        self._device = device
        self._pbc = pbc
        self._only_theta = False
        self._only_phi = False
        self._inverse_symmetry = inverse_symmetry

        if lattice_shape == 'sq':
            """
            tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    ])
            """
            self.filter = torch.tensor([
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)
        else:
            """
            tensor([
                    [0, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0],
                    ])

            NNNN:
            tensor([
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    ])
            """
            self.filter = torch.tensor([
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)
        
        self.filter_num = self.filter.sum().cpu().item()
        self.filter_num *= 1 if custom_filter else 0

        # self.model = mlp_cnn(state_size=state_size, K=K, F=F, stride0=stride0, 
        #         stride=stride, relu_type=relu_type, 
        #         pbc=pbc, bias=bias, momentum=momentum, groups=1, alpha=alpha,
        #         custom_filter=custom_filter, lattice_shape=lattice_shape, 
        #         device=device)
        self.model_phi = mlp_cnn_single(state_size=state_size, K=K, F=F, stride=stride, pbc=pbc,
                        custom_filter=False, lattice_shape=lattice_shape, device=device, 
                        output_mode='phi', inverse_symmetry=self._inverse_symmetry)

        self.model_theta = mlp_cnn_single(state_size=state_size, K=K, 
                        F=F, stride=stride, pbc=pbc, custom_filter=False, lattice_shape=lattice_shape, 
                        device=device, output_mode='theta', inverse_symmetry=self._inverse_symmetry)

        # self.model_theta = mlp_cnn_single(state_size=state_size, K=state_size[-2], 
        #                 F=[8,8], stride=stride, pbc=pbc, alpha=alpha,
        #                 custom_filter=custom_filter, lattice_shape=lattice_shape, device=device, 
        #                 act_require_grad=False, output_mode='theta')

        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.ones_(m.weight)
                #nn.init.ones_(m.weight)
            elif isinstance(m, P4MConvP4M):
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, P4MConvZ2):
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=1/np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.model_phi.apply(weight_init)
        #if custom_filter:
        #    complex_init(self.model_phi, self.filter)
        self.model_phi = self.model_phi.to(precision).to(device)

        self.model_theta.apply(weight_init)
        self.model_theta = self.model_theta.to(precision).to(device)

    def symmetry(self, x):
        sym_N = 1
        for func in self.sym_funcs:
            x, n = func(x)
            sym_N *= n
        return x, sym_N

    # apply symmetry
    def forward(self, x):
        # shape of x: (batch_size, Dp, L, W)
        if self._inverse_symmetry:
            x = x.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])

        if self._only_phi:
            logphi = self.model_phi(x.clone())
            theta = torch.zeros_like(logphi)
        else:
            theta = self.model_theta(x.clone())
            if self._only_theta:
                logphi = torch.zeros_like(theta)
            else:
                logphi = self.model_phi(x.clone())
        
        return torch.stack((logphi, theta), dim=1)

def mlp_cnn_sym(state_size, K, F=[4,3,2], stride=[1], device="cpu",
                pbc=True, sym_funcs=[identity], alpha=0.35,
                custom_filter=False, lattice_shape=None, inverse_symmetry=False, precision=torch.float32):
    return sym_model(state_size=state_size, K=K, F=F, stride=stride, device=device, pbc=pbc, 
                sym_funcs=sym_funcs, alpha=alpha, custom_filter=custom_filter, 
                lattice_shape=lattice_shape, inverse_symmetry=inverse_symmetry, precision=precision)

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 286
    torch.manual_seed(seed)
    np.random.seed(seed)
    logphi_model = mlp_cnn_sym([10, 10, 2], 5, [4,4,4,4], stride=[1], 
            pbc=True, custom_filter=False, lattice_shape='sq', inverse_symmetry=False, precision=torch.float64)
    #op_model = mlp_cnn([10,10,2], 2, [2],complex_nn=True, output_size=2, relu_type='sReLU', bias=True)
    print(logphi_model)
    print(get_paras_number(logphi_model))
    import sys
    sys.path.append('..')
    from ops.HS_spin2d import get_init_state
    #from ops.tfim_spin1d import get_init_state
    state0,_ = get_init_state([10,10,2], kind='rand', n_size=10000)
    #state_zero = torch.from_numpy(state0[0][None,...])
    # state_zero = torch.stack((state_zero, torch.zeros_like(state_zero)), dim=1)
    # state_t0 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=4, dims=[2,3], center=[1])
    # t1 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=3, dims=[2,3], center=[1])
    # t2 = rot60(t1, num=1, dims=[2,3], center=[1])
    #print(state0[0].shape)
    #print(state_t0)
    #print(t2 - state_t0)
    # print(complex_periodic_padding(state_zero, [3,3], [1,1], dimensions='2d'))
    #print(state0.shape)
    import time
    tic = time.time()
    #state0 = torch.from_numpy(state0).float()
    #state0r = torch.transpose(state0, 2, 3)
    #tt2 = logphi_model(state0r).detach().numpy()
    #print(np.max(tt - tt2))
    #state_unique = np.unique(state0, return_inverse=True, return_counts=True, axis=0)
    #state0 = torch.from_numpy(state0).float()
    #tt = logphi_model(state0).detach().numpy()[:,0]
    #state_unique, inverse_indices, counts = torch.unique(state0, return_inverse=True, return_counts=True, dim=0)
    #print(time.time() - tic)
    #print(tt[inverse_indices].shape, state_unique.shape)
    #logphi_model._inverse_symmetry = True
    tic = time.time()
    state0 = torch.from_numpy(state0).double()
    # #print(state0[0])
    # #print(torch.flip(state0[0], dims=[2]))
    # # state0r = torch.flip(state0, dims=[1])
    # #state0r = torch.roll(state0, shifts=1, dims=2)
    # #state0 = torch.flip(state0, dims=[3])
    # state0r = torch.flip(state0, dims=[1])
    # print(state0.shape)
    #print(state0[1, 0] - state0[1, 1])
    #state0r = rot60(state0, 1, dims=[2,3], center=[0])
    #state0r = torch.transpose(state0, 2, 3)
    #logphi_model = logphi_model.double()
    # logphi_model._only_theta = False
    tt1 = logphi_model(state0)
    # print(tt1.shape)
    # print(time.time() - tic)
    # # #print(logphi_model.model.forward.__dir__())
    # tt2 = logphi_model(state0r)
    #print(tt2[0])
    print(tt1)
    # # tt1.sum().backward()
    for name, p in logphi_model.named_parameters():
        print(name, p.shape)

    # # state0x = logphi_model.pick_sym_config_new(state0[0][None,...])
    # # print(state0[0])
    # # print(state0x)
    # # print(0==torch.max(tt1 - tt2))
    # #print(tt1)
    # print((tt1))
    # #print(logphi_model(state0[0][None,...]))
    # print(time.time() - tic)

    #x = logphi_model.pick_sym_config(state0)
    #print(x.shape)

    # from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
    # # Construct G-Conv layers
    # C1 = P4MConvZ2(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1)
    # C2 = P4MConvP4M(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)
        
    # #x = torch.autograd.Variable(torch.randn(10, 3, 9, 9))
    # #print(x.shape)
    # y1 = C2(C1(state0))
    # y2 = C2(C1(state0r))
    # print(y1.sum(dim=[2,3,4])[0])
    # print(y2.sum(dim=[2,3,4])[0])

    # # print(isinstance(logphi_model, conv_re))
    # for name,p in logphi_model.named_parameters():
    #     # if p.requires_grad:
    #     print(name, p.shape)
    # for name, p in logphi_model.named_parameters():
    #    print(name)
    # param_grp = []
    # for idx, m in enumerate(logphi_model.modules()):
    #     if isinstance(m, ComplexConv):
    #         param_grp.append(m.conv_re.weight)

    # print(dir(logphi_model.model))

    # x, _ = logphi_model.symmetry(state0[0][None,...])
    # print(x)
    # # print(sort_by_label2d(x[None,...], x.shape[-2], x.shape[-1]))

    # x, _ = logphi_model.symmetry(state0r[0][None,...])
    # print(x)
    # print(sort_by_label2d(x[None,...], x.shape[-2], x.shape[-1]))

    #import scipy.io as sio
    #sio.savemat('test2.mat', {'tt2':tt})
    # print(logphi_model(torch.from_numpy(state0).float())[:3])
    #state_t = torch.roll(state_t0, shifts=2, dims=2)
    # state_t = torch.rot90(torch.from_numpy(state0[0][None,...]).float(),2, dims=[2,3])
    #print(state_t)
    #print(logphi_model(state_t))
    #logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [2,2], stride=[1], complex_nn=True,
    #                       output_size=2, relu_type='selu', bias=True, momentum=[0,0], sym_funcs=[identity])
    #logphi_model_sym.load_state_dict(logphi_model.state_dict())
    # logphi_model_sym.eval()
    #print(logphi_model_sym(torch.from_numpy(state0).float())[1])
    # print(list(logphi_model(torch.from_numpy(state0).float()).size()))
    # x, M = reflection(torch.from_numpy(state0))

    #pad = (0,0,0,2)
    #x = torch.nn.functional.pad(torch.from_numpy(state0[0][None,...]).float(), pad, mode='circular', value=0)
    #print(x)
    #print(x.shape)
    #print(x.shape)
    #print(complex_periodic_padding(torch.from_numpy(state0[0]).reshape(1,2,1,4,4),[2,2],'2d'))

    # phi = logphi_model(torch.from_numpy(state0).float())
    # # logphi = phi[:,0].reshape(1,-1)
    # # theta = phi[:,1].reshape(1,-1)
    # print(phi[:,0].std()/phi[:,0].mean())
    # print(phi[:,1].std()/phi[:,1].mean(),phi[:,1].max(), phi[:,1].min())

    # logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [3,2],complex_nn=True,
    #                        output_size=2, relu_type='softplus2', bias=True)
    # phi = logphi_model_sym(torch.from_numpy(state0).float())
    # print(get_paras_number(logphi_model_sym))
