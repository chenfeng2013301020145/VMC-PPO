# encoding: utf-8

import numpy as np

class updator():
    def __init__(self, state_size):
        self._dimensions = len(state_size) - 1
        if self._dimensions == 1: 
            self._N = state_size[0]
        else:
            self._L = state_size[0]
            self._W = state_size[1]
        self._Dp = state_size[-1]

    def onehot2value(self, state): 
        if self._dimensions == 1:
            state_v = np.arange(0,self._Dp).reshape(self._Dp,1)*np.squeeze(state)
        else:
            state_v = np.arange(0,self._Dp).reshape(self._Dp,1,1)*np.squeeze(state)
        return np.sum(state_v,0).astype(dtype=np.int8)
    
    def value2onehot(self, state):
        if self._dimensions == 1:
            state_onehot = np.zeros([self._Dp, self._N])
            state_onehot[state.astype(dtype=np.int8), range(self._N)] = 1
        else:
            X, Y = np.meshgrid(range(self._L), range(self._W))
            state_onehot = np.zeros([self._Dp, self._L, self._W])
            state_onehot[state.astype(dtype=np.int8), Y, X] = 1
        return state_onehot

    def generate_mask(self, n_sample):
        # for value encoding
        # return np.random.randint(2, size=[n_sample, self._N])
        # flip single bit:
        if self._dimensions == 1:
            masks = np.zeros([n_sample, self._N])
            pos = np.random.randint(self._N, size=n_sample)
            masks[range(n_sample),pos] = 1
        else:
            masks = np.zeros([n_sample, self._L, self._W])
            pos_x = np.random.randint(self._L, size=n_sample)
            pos_y = np.random.randint(self._L, size=n_sample)
            masks[range(n_sample),pos_x, pos_y] = 1
        return masks.astype(np.int8)

    def _get_update(self, state, mask):
        if self._Dp != 1:
            # convert to value encoding
            self._state = self.onehot2value(state)
            self._state = np.bitwise_xor(self._state, mask)
            # convert to onehot encoding
            return self.value2onehot(self._state)
        else:
            self._state = np.bitwise_xor(self._state, mask)
            return self._state

if __name__=='__main__':
    from tfim_spin2d import get_init_state

    state_size = [4,4,2]
    state0 = get_init_state(state_size, kind='rand')
    print(state0)

    Op = updator(state_size)
    state_v = Op.onehot2value(state0)
    print(state_v)

    state_onehot = Op.value2onehot(state_v)
    print(state_onehot)

    masks = Op.generate_mask(100)
    print(masks[10])
    print(Op._get_update(state_onehot, masks[10]) - state_onehot)