# encoding: utf-8
"""
Two dimensional Heisenberg model on 
square lattice, triangle lattice.
"""

import numpy as np

def get_init_state(state_size, kind='rand', n_size=1):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
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
            state_v = np.zeros([L, W])
            pos = np.random.choice(L*W, L*W//2, replace=False)
            pos_y = pos // W
            pos_x = pos % W
            state_v[pos_y, pos_x] = 1
            # print(state_v.sum())
            state[i] = value2onehot(state_v, Dp)
            state_v_r += (state_v - (Dp-1)/2).sum()

    return state, state_v_r

def value2onehot(state, Dp):
    L = state.shape[0]
    W = state.shape[1]
    X, Y = np.meshgrid(range(W), range(L))
    state_onehot = np.zeros([Dp, L, W])
    state_onehot[state.astype(dtype=np.int8), Y, X] = 1
    return state_onehot

def _find_states(state, update_size, nearest_neighbors_j1, nearest_neighbors_j2=None, j2=0, pbc=True, marshall_sign=1):
    # shape: (Dp, L, W)
    L = state.shape[-2]
    W = state.shape[-1]
    Dp = state.shape[0]
    states = np.zeros([update_size, Dp, L, W])
    coeffs = np.zeros(update_size)
    diag = 0.0
    cnt = 0
    for r in range(L):
        for c in range(W):
            # j1
            for dr, dc in nearest_neighbors_j1:
                rr, cc = r + dr, c + dc
                if rr >= L or cc >= W:
                    if pbc:
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
                    coeffs[cnt] = 0.5*marshall_sign
                    diag -= 0.25
                    cnt += 1
                else:
                    diag += 0.25
                    
            # j2
            if j2 != 0:
                for dr, dc in nearest_neighbors_j2:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if pbc:
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
                        coeffs[cnt] = 0.5*j2
                        diag -= 0.25*j2
                        cnt += 1
                    else:
                        diag += 0.25*j2

    states[cnt] = state.copy()
    coeffs[cnt] = diag
    return states, coeffs, cnt+1

class Heisenberg2DSquare():

    def __init__(self, state_size, pbc=True, Marshall_sign=False):
        """Initializes a 2D Heisenberg AFM Hamiltonian.

        H = \sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j
          = \sum_<i,j>1/2(S^+_iS^-_j + h.c) + S^z_iS^z_j
        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._nearest_neighbors = ((0, 1), (1, 0))
        self._update_size = 2*state_size[0]*state_size[1] + 1
        self._Marshall_sign = -1 if Marshall_sign else 1

    def find_states(self, state: np.ndarray):
        # shape: (Dp, L, W)
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)
        diag = 0.0
        cnt = 0
        oy, ox, sy, sx = [],[],[],[]
        for r in range(L):
            for c in range(W):
                for dr, dc in self._nearest_neighbors:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        oy.append(r)
                        ox.append(c)
                        sy.append(rr)
                        sx.append(cc)
                        coeffs[cnt] = self._Marshall_sign*0.5
                        diag -= 0.25
                        cnt +=1
                    else:
                        diag += 0.25
                        
        temp = np.repeat(state.reshape(1,Dp,L,W), cnt, axis=0)
        temp[range(cnt), :, [oy, sy], [ox, sx]] = temp[range(cnt), :, [sy, oy], [sx, ox]]
        states[:cnt] = temp
        states[cnt] = state.copy()
        coeffs[cnt] = diag
        return states, coeffs, cnt+1

class Heisenberg2DTriangle():

    def __init__(self, state_size, pbc=True):
        """Initializes a 2D Heisenberg AFM Hamiltonian.

        H = \sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j
          = \sum_<i,j>1/2(S^+_iS^-_j + h.c) + S^z_iS^z_j
        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._nearest_neighbors = ((0, 1), (1, 0), (1, 1))
        self._update_size = 3*state_size[0]*state_size[1] + 1

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
                for dr, dc in self._nearest_neighbors:
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
                        diag -= 0.25
                        cnt += 1
                    else:
                        diag += 0.25
        states[cnt] = state.copy()
        coeffs[cnt] = diag
        return states, coeffs, cnt+1
    
class J1J2_2DSquare():

    def __init__(self, state_size, pbc=True, j2=0, Marshall_sign=False):
        """Initializes a 2D J1-J2 AFM Hamiltonian.

        H =   J1*(\sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j)
            + J2*(\sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j)
          =   J1*(\sum_<i,j>1/2(S^+_iS^-_j + h.c) + S^z_iS^z_j)
            + J2*(\sum_<i,j>1/2(S^+_iS^-_j + h.c) + S^z_iS^z_j)
        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._j2 = j2
        self._nearest_neighbors_j1 = ((0, 1), (1, 0))
        self._nearest_neighbors_j2 = ((1, 1), (-1, 1))
        self._update_size = 4*state_size[0]*state_size[1] + 1
        self._Marshall_sign = -1 if Marshall_sign else 1

    def find_states(self, state: np.ndarray):
        return _find_states(state, 
                            update_size=self._update_size, 
                            nearest_neighbors_j1=self._nearest_neighbors_j1, 
                            nearest_neighbors_j2=self._nearest_neighbors_j2, 
                            j2=self._j2, pbc=self._pbc,
                            marshall_sign=self._Marshall_sign)
        # # shape: (Dp, L, W)
        # L = state.shape[-2]
        # W = state.shape[-1]
        # Dp = state.shape[0]
        # states = np.zeros([self._update_size, Dp, L, W])
        # coeffs = np.zeros(self._update_size)
        # diag = 0.0
        # cnt = 0
        # for r in range(L):
        #     for c in range(W):
        #         # j1
        #         for dr, dc in self._nearest_neighbors_j1:
        #             rr, cc = r + dr, c + dc
        #             if rr >= L or cc >= W:
        #                 if self._pbc:
        #                     rr %= L
        #                     cc %= W
        #                 else:
        #                     continue
        #             if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
        #                 temp = state.copy()
        #                 # This is the correct way of swapping states when
        #                 # temp.ndim > 2.
        #                 temp[:, [r, rr], [c, cc]] = temp[:, [rr, r], [cc, c]]
        #                 states[cnt] = temp
        #                 coeffs[cnt] = self._Marshall_sign*0.5
        #                 diag -= 0.25
        #                 cnt += 1
        #             else:
        #                 diag += 0.25
                        
        #         # j2
        #         if self._j2 != 0:
        #             for dr, dc in self._nearest_neighbors_j2:
        #                 rr, cc = r + dr, c + dc
        #                 if rr >= L or cc >= W:
        #                     if self._pbc:
        #                         rr %= L
        #                         cc %= W
        #                     else:
        #                         continue
        #                 if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
        #                     temp = state.copy()
        #                     # This is the correct way of swapping states when
        #                     # temp.ndim > 2.
        #                     temp[:, [r, rr], [c, cc]] = temp[:, [rr, r], [cc, c]]
        #                     states[cnt] = temp
        #                     coeffs[cnt] = 0.5*self._j2
        #                     diag -= 0.25*self._j2
        #                     cnt += 1
        #                 else:
        #                     diag += 0.25*self._j2

        # states[cnt] = state.copy()
        # coeffs[cnt] = diag
        # return states, coeffs, cnt+1

if __name__ == '__main__':
    state0, _ = get_init_state([4,4,2], kind='rand', n_size=10)
    print(state0[1])
    
    state0 = np.array([[[0,1],[1,0]],[[1,0],[0,1]]])
    _ham = Heisenberg2DTriangle([2,2,2], pbc=True)
    ustates, ucoeffs = _ham.find_states(state0)
    print(ucoeffs)
    print(ustates)