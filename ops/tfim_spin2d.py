# encoding: utf-8

import numpy as np 

def get_init_state(state_size, kind='rand', n_size=1):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    state = np.zeros([n_size, Dp, L, W])

    if kind == 'ferro':
        for i in range(n_size):
            pos = np.random.randint(0, Dp)
            state[i,pos,:,:] = 1

    if kind == 'rand':
        state_v = np.random.randint(0, Dp, size=(n_size, L, W))
        X, Y = np.meshgrid(range(L), range(W))
        for i in range(n_size):
            state[i,state_v[i],Y,X] = 1

    return state, 0

def onehot2value(state, Dp): 
    state_v = np.arange(0,Dp).reshape(Dp,1,1)*np.squeeze(state)
    return np.sum(state_v,0).astype(dtype=np.int8)
    
class TFIMSpin2D():

    def __init__(self, g: float, state_size, pbc=True):
        """
        Tranverse Ising model in 2 dimension:
        Hamiltonian: 
            H = - sum_ij{sigma^z_ij * sigma^z_{i + 1}j + sigma^z_ij * sigma^z_i{j + 1}} 
                + g*sum_ij{sigma^x_ij}

        Args:
            g: Strength of the transverse field.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._g = g
        self._update_size = state_size[0]*state_size[1] + 1

    def find_states(self, state: np.ndarray):
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([L*W+1, Dp, L, W])
        coeffs = np.zeros(L*W+1)

        # off-diagnal
        cnt = 0
        for y in range(L):
            for x in range(W):
                temp = state.copy()
                temp[0,x,y], temp[1,x,y] = state[1,x,y], state[0,x,y]
                states[cnt] = temp
                coeffs[cnt] = 1/2*self._g
                cnt += 1

        # diagnal 
        state_v = onehot2value(state, Dp) - 1/2
        state_l = np.concatenate((state_v, state_v[:,0].reshape(L,1)), axis=1)
        state_r = np.concatenate((state_v[:,-1].reshape(L,1), state_v), axis=1)
        state_u = np.concatenate((state_v, state_v[0].reshape(1,W)), axis=0)
        state_d = np.concatenate((state_v[-1].reshape(1,W), state_v), axis=0)
        if self._pbc:   
            diag = - np.sum(state_l[:,1:]*state_r[:,1:]) - np.sum(state_u[1:]*state_d[1:])
        else:
            diag = - np.sum(state_l[:,1:-1]*state_r[:,1:-1]) - np.sum(state_u[1:-1]*state_d[1:-1])
            
        states[-1] = state
        coeffs[-1] = diag

        return states, coeffs

if __name__=='__main__':
    from tfim_spin2d import get_init_state

    state_size = [10,10,2]
    state0 = get_init_state(state_size, kind='rand',n_size=10)
    print(state0[0])
    # print(np.sum(state0[0],0))
    '''
    state_v = onehot2value(state0[0],Dp=2)
    print(state_v)
    
    ham = TFIMSpin2D(1, pbc=True)
    states, coeffs = ham.find_states(np.squeeze(state0[2]))
    print(states)
    print(coeffs)
    '''
    