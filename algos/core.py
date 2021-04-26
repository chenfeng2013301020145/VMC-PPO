# encoding: utf-8

import torch.nn as nn
import torch
import numpy as np
# from utils import extract_weights,load_weights

def periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of x: (batch_size, Dp, N)
        return torch.cat((x, x[:,:,0:kernel_size-1]), -1)
    else:
        # shape of real x: (batch_size, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,0:kernel_size[0]-1,:]), -2)
        x = torch.cat((x, x[:,:,:,0:kernel_size[1]-1]), -1)
        return x

def complex_periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of complex x: (batch_size, 2, Dp, N)
        return torch.cat((x, x[:,:,:,0:kernel_size-1]), -1)
    else:
        # shape of complex x: (batch_size, 2, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,:,0:kernel_size[0]-1,:]), -2)
        x = torch.cat((x, x[:,:,:,:,0:kernel_size[1]-1]), -1)
        return x
    
def sym_padding(x, dimensions):
    if dimensions == '1d':
        # shape of complex x: (batch_size, Dp, N)
        return torch.cat((x, x[:,:,0:1]), -1)
    else:
        # shape of complex x: (batch_size, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,0:1,:]), -2)
        x = torch.cat((x, x[:,:,:,0:1]), -1)
        return x

def complex_sym_padding(x, dimensions):
    if dimensions == '1d':
        # shape of complex x: (batch_size, 2, Dp, N)
        return torch.cat((x, x[:,:,:,0:1]), -1)
    else:
        # shape of complex x: (batch_size, 2, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,:,0:1,:]), -2)
        x = torch.cat((x, x[:,:,:,:,0:1]), -1)
        return x

def get_paras_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def gradient(y, x, grad_outputs=None):
    """compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def complex_init(model):
    cnt = -1
    for name, p in model.named_parameters():
        if int(name.split(".")[0]) - cnt == 1:
            #rho = np.random.rayleigh(1, size=p.data.shape)
            #rho = torch.from_numpy(np.random.uniform(0., 1., size=p.data.shape)).float()
            rho = torch.ones_like(p.data)
            theta = np.random.uniform(-np.pi, +np.pi, size=p.data.shape)
            w = np.sqrt(np.prod(p.data.shape[:-2])*(p.data.shape[-1] + p.data.shape[-2]))
            cnt += 1
        if name.split(".")[-1] == 'weight':
            if name.split(".")[3] == 'conv_re' or name.split(".")[2] == 'linear_re':
                p.data = torch.from_numpy(np.cos(theta)).float()*rho/w
            elif name.split(".")[3] == 'conv_im' or name.split(".")[2] == 'linear_im':
                p.data = torch.from_numpy(np.sin(theta)).float()*rho/w
    return

# Real CNN 
#--------------------------------------------------------------------
class CNN_layer(nn.Module):
    def __init__(self,K,F_in,F_out,act=nn.ReLU,pbc=True,dimensions='1d'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN_layer,self).__init__()
        self._pbc = pbc
        self.dimensions = dimensions
        self.K = [K,K] if type(K) is int else K
        if dimensions == '1d':
            self.conv = nn.Sequential(nn.Conv1d(F_in,F_out,self.K,1,0),act())
        else:
            self.conv = nn.Sequential(nn.Conv2d(F_in,F_out,self.K,[1,1],0),act())
            
    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions=self.dimensions)
        x = self.conv(x)
        return x

class OutPut_layer(nn.Module):
    def __init__(self,K,F,output_size, pbc=True, dimensions='1d'):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut_layer,self).__init__()
        self._pbc = pbc
        self.output_size = output_size
        self.dimensions = dimensions
        self.K = [K,K] if type(K) is int else K
        self.linear = nn.Linear(F,output_size, bias=False)
            
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions=self.dimensions)
        x = x.sum(2) if self.dimensions == '1d' else x.sum(dim=[2,3])
        return self.linear(x)

# COMPLEX NEURAL NETWORK
# ----------------------------------------------------------------
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                    padding=0, dilation=1, groups=1, bias=True, dimensions='1d'):
        super(ComplexConv,self).__init__()

        ## Model components
        if dimensions == '1d':
            self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
        return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imaginary = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
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
    def __init__(self, relu_type='zReLU'):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexReLU, self).__init__()
        self.type = relu_type
        self._threshold = -0.1

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        if self.type == 'zReLU':
            mask = ((0 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'sReLU':
            mask = ((-np.pi/2 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'modReLU':
            modulus = torch.clamp(z.abs(), min=1e-5)
            z *= torch.relu(1. - self._threshold / modulus)
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus':
            z = torch.log(1. + torch.exp(z))
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus2':
            z = torch.log(1./2. + torch.exp(z)/2.)
            return torch.stack((z.real, z.imag), dim=1)
        else:
            return torch.stack((torch.relu(real), torch.relu(imag)), dim=1)

# COMPLEX CNN
#--------------------------------------------------------------------
class CNN_complex_layer(nn.Module):
    def __init__(self,K,F_in,F_out,layer_name='mid',relu_type='sReLU',
                 pbc=True, bias=True, dimensions='1d'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN_complex_layer,self).__init__()
        self.K = [K,K] if type(K) is int else K
        self._pbc = pbc
        self.layer_name = layer_name
        self.dimensions = dimensions
        
        complex_relu = ComplexReLU(relu_type)
        complex_conv = ComplexConv(F_in,F_out,self.K,1,0, dimensions=dimensions, bias=bias)
        self.conv = nn.Sequential(*[complex_conv, complex_relu])

    def forward(self, x):
        if self.layer_name == '1st':
            x = torch.stack((x, torch.zeros_like(x)), dim=1)
        if self._pbc:
            x = complex_periodic_padding(x, self.K, dimensions=self.dimensions)
        x = self.conv(x)
        return x

class OutPut_complex_layer(nn.Module):
    def __init__(self,K,F,pbc=True,dimensions='1d'):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut_complex_layer,self).__init__()
        self.K = [K,K] if type(K) is int else K
        self.F = F
        self._pbc=pbc
        self.dimensions = dimensions
        self.linear = ComplexLinear(F,1, bias=False)
    
    def forward(self,x):
        x = complex_sym_padding(x, dimensions=self.dimensions)
        # shape of complex x: (batch_size, 2, F, N) or (batch_size, 2, F, L, W)
        # norm = np.sqrt(np.prod(x.shape[3:]))
        x = x.sum(3) if self.dimensions=='1d' else x.sum(dim=[3,4])
        x = self.linear(x).squeeze(-1)
        x[:,1] = torch.fmod(x[:,1], 2*np.pi) - np.pi
        return x
    
#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F=[4,3,2], output_size=1, output_activation=False, act=nn.ReLU,
        complex_nn=False, inverse_sym=False, relu_type='sReLU', pbc=True, bias=True):
    dim = len(state_size) - 1
    dimensions = '1d' if dim == 1 else '2d'
    layers = len(F)
    name_index = 0
    if complex_nn:
        Dp = state_size[-1]

        input_layer = CNN_complex_layer(K=K, F_in=Dp, F_out=F[0], layer_name='1st', dimensions=dimensions,
                                        relu_type=relu_type, pbc=pbc, bias=bias)
        output_layer = OutPut_complex_layer(K,F[-1], pbc=pbc, dimensions=dimensions)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN_complex_layer(K=K, F_in=F[i-1], F_out=F[i], dimensions=dimensions,
                            relu_type=relu_type, pbc=pbc, bias=bias) for i in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, ComplexConv):
                nn.init.xavier_uniform_(m.conv_re.weight, gain=1/np.sqrt(2))
                nn.init.xavier_uniform_(m.conv_im.weight, gain=1/np.sqrt(2))
                if m.conv_re.bias is not None:
                    nn.init.zeros_(m.conv_re.bias)
                    nn.init.zeros_(m.conv_im.bias)
            elif isinstance(m, ComplexLinear):
                nn.init.xavier_uniform_(m.linear_re.weight, gain=1/np.sqrt(2))
                nn.init.xavier_uniform_(m.linear_im.weight, gain=1/np.sqrt(2))

        model = nn.Sequential(*cnn_layers)
        model.apply(weight_init)
    else:
        Dp = state_size[-1]

        input_layer = CNN_layer(K=K, F_in=Dp, F_out=F[0], act=act, pbc=pbc, dimensions=dimensions)
        output_layer = OutPut_layer(K,F[-1],output_size,pbc=pbc, dimensions=dimensions)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN_layer(K=K, F_in=F[i-1], F_out=F[i], dimensions=dimensions,
                                act=act, pbc=pbc) for i in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

        model = nn.Sequential(*cnn_layers)
        model.apply(weight_init)
    return model, name_index

# physical symmetry 
# -------------------------------------------------------------------------------------------------
class sym_model(nn.Module):
    def __init__(self, state_size, K, F=[4,3,2], output_size=1, output_activation=False, act=nn.ReLU,
        complex_nn=False, relu_type='sReLU', pbc=True, bias=True):
        super(sym_model,self).__init__()
        self.model, _ = mlp_cnn(state_size=state_size, K=K, F=F, output_size=output_size, output_activation=output_activation, 
                    act=act, complex_nn=complex_nn, relu_type=relu_type, pbc=pbc, bias=bias)
        
    def forward(self,x):
        # input shape: (batch, Dp, N) or (batch, Dp, L, W)
        x_inverse = torch.flip(x, dims=[1])
        norm = 2
        z = (self.model(x) + self.model(x_inverse))/norm
        x[:,0] = z[:,0]
        x[:,1] = x[:,1]
        return x

def mlp_cnn_sym(state_size, K, F=[4,3,2], output_size=1, output_activation=False, act=nn.ReLU,
        complex_nn=False, relu_type='sReLU', pbc=True, bias=True):
    model = sym_model(state_size=state_size, K=K, F=F, output_size=output_size, output_activation=output_activation, 
                    act=act, complex_nn=complex_nn, relu_type=relu_type, pbc=pbc, bias=bias)
    name_index = 1
    return model, name_index

    
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 10086
    torch.manual_seed(seed)
    np.random.seed(seed)
    logphi_model, _ = mlp_cnn([3,3,2], 2, [3,2],complex_nn=True,
                           output_size=2, relu_type='softplus2', bias=True)
    #op_model = mlp_cnn([10,10,2], 2, [2],complex_nn=True, output_size=2, relu_type='sReLU', bias=True)
    # print(logphi_model)
    print(get_paras_number(logphi_model))
    import sys
    sys.path.append('..')
    from ops.HS_spin2d import get_init_state
    state0,_ = get_init_state([3,3,2], kind='rand', n_size=500)
    print(state0[0]) 
    print(torch.flip(torch.from_numpy(state0[0]), dims=[1,2]))
    #print(complex_periodic_padding(torch.from_numpy(state0[0]).reshape(1,2,1,4,4),[2,2],'2d'))

    phi = logphi_model(torch.from_numpy(state0).float())
    # logphi = phi[:,0].reshape(1,-1)
    # theta = phi[:,1].reshape(1,-1)
    print(phi[:,0].std()/phi[:,0].mean())
    print(phi[:,1].std()/phi[:,1].mean(),phi[:,1].max(), phi[:,1].min())
    
    logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [3,2],complex_nn=True,
                           output_size=2, relu_type='softplus2', bias=True)
    phi = logphi_model_sym(torch.from_numpy(state0).float())
    print(get_paras_number(logphi_model_sym))

