# encoding: utf-8
"""
Two dimensional Heisenberg model on Kagome lattice.
"""

import numpy as np

def get_init_state(state_size, kind='rand', n_size=1):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    
    L3 = L // 2
    W3 = W // 2
    num = L3*W3*3
    state = np.zeros([n_size, Dp, L, W])
    state_v_r = 0

    if kind == 'neel':
        for i in range(n_size):
            state_v = np.zeros(L*W)
            state_v[np.arange(1, L*W, 2)] = 1
            # print(state_v.sum())
            state[i] = value2onehot(state_v.reshape(L,W), Dp)
            state_v_r += (state_v - (Dp-1)/2).sum()

    if kind == 'rand':
        for i in range(n_size):
            state_v = np.zeros([L3, W3, 3])
            pos = np.random.choice(num, num//2, replace=False)
            pos_in = pos % 3
            pos_out = pos // 3
            pos_y = pos_out // W3
            pos_x = pos_out % W3
            state_v[pos_y, pos_x, pos_in] = 1
            # print(state_v.sum())
            state[i] = state3_to_state2(value2onehot(state_v, Dp), state_size)
            state_v_r += (state_v - (Dp-1)/2).sum()

    return state, state_v_r

def state3_to_state2(state3, state_size):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    state2 = np.zeros([Dp, L, W])
    
    L3, W3 = state3.shape[0], state3.shape[1]
    X, Y = np.meshgrid(range(W3), range(L3))
    for x,y in zip(X.reshape(-1), Y.reshape(-1)):
        state2[:, 2*y, 2*x] = state3[:, y, x, 0]
        state2[:, 2*y, 2*x + 1] = state3[:, y, x, 1]
        state2[:, 2*y + 1, 2*x] = state3[:, y, x, 2]
    return state2

def state2_to_state3(state2, state_size):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    
    L3 = L // 2
    W3 = W // 2
    state3 = np.zeros([Dp, L3, W3, 3])
    X, Y = np.meshgrid(range(W3), range(L3))
    
    for x,y in zip(X.reshape(-1), Y.reshape(-1)):
        state3[:, y, x, :] = state2[:, 2*y:2*y+2, 2*x:2*x+2].reshape(Dp, 4)[:, :3]
    return state3

def value2onehot(state, Dp):
    L = state.shape[0]
    W = state.shape[1]
    I = state.shape[2]
    X, Y, Z = np.meshgrid(range(W), range(L), range(I))
    state_onehot = np.zeros([Dp, L, W, I])
    state_onehot[state.astype(dtype=np.int8), Y, X, Z] = 1
    return state_onehot


class Heisenberg2DKagome():

    def __init__(self, state_size, pbc=True):
        """Initializes a 2D Heisenberg AFM Hamiltonian.

        H = \sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j
          = \sum_<i,j>1/2(S^+_iS^-_j + h.c) + S^z_iS^z_j
        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._00_nn = ((1, 0), (0, 1))
        self._01_nn = ((0, 1), (1, 1))
        self._10_nn = ((1, 0), (1, 1))
        self._11_nn = ((0, 0), (0, 0))
        self._update_size = 2*state_size[0]*state_size[1] + 1

    def find_states(self, state: np.ndarray):
        # shape: (Dp, L, W)
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)
        diag = 0.0
        cnt = 0
        for r in range(L):
            for c in range(W):
                fac = 1
                if r%2 == 0 and c%2 == 0:
                    nearest_neighbors = self._00_nn
                elif r%2 == 0 and c%2 == 1:
                    nearest_neighbors = self._01_nn
                elif r%2 == 1 and c%2 == 0:
                    nearest_neighbors = self._10_nn
                elif r%2 == 1 and c%2 == 1:
                    nearest_neighbors = self._11_nn
                    fac = 0
                
                for dr, dc in nearest_neighbors:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        temp = state.copy()

                        # This is the correct way of swapping states when
                        # temp.ndim > 2.
                        temp[:, [r, rr], [c, cc]] = temp[:, [rr, r], [cc, c]]
                        states[cnt] = temp
                        coeffs[cnt] = 0.5
                        diag -= 0.25*fac
                    else:
                        diag += 0.25*fac
                    cnt +=1
        states[-1] = state.copy()
        coeffs[-1] = diag
        return states, coeffs
    
    
if __name__ == '__main__':
    import torch
    k = torch.tensor([0,1,2])
    print(k)
    state3 = k.repeat(2,2,2,1)
    print(state3.shape)
    state2 = state3_to_state2(state3, [4,4,2])
    print(state2)
    state4 = state2_to_state3(state2, [4,4,2])
    print(torch.tensor(state4) - state3)
    state, ms = get_init_state([4,4,2])
    print(ms)