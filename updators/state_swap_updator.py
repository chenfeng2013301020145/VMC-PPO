# encoding: utf-8
# compatible with 2d system

import numpy as np

class updator():
    def __init__(self, state_size):
        if len(state_size) - 1 == 1:
            self._N = state_size[0]
        else:
            self._N = state_size[0]*state_size[1]
        self._Dp = state_size[-1]

    def generate_mask(self, n_sample):
        rang = range(n_sample)
        swaps = np.random.randint(0, self._N, (2, n_sample))
        masks = np.arange(self._N)[None, :].repeat(n_sample, axis=0)
        masks[rang, swaps[0]], masks[rang, swaps[1]] = (
            masks[rang, swaps[1]], masks[rang, swaps[0]])
        # print(np.random.randint(0,10))
        return masks

    def _get_update(self, state, mask):
        temp = state.reshape(self._Dp, self._N).T
        state_f = temp[mask]
        # self._state = np.concatenate((self._state, self._state[0,:].reshape(1,self._Dp)),0)
        return state_f.T.reshape(state.shape)

if __name__ == "__main__":
    from HS_spin2d import get_init_state
    state0, _ = get_init_state([4,3,2], kind='rand', n_size=10)
    print(state0[2].shape)
    
    Update = updator([4,3,2])
    masks = Update.generate_mask(100)
    print(masks[10])
    
    print(state0[2])
    statef = Update._get_update(state0[2], masks[10])
    print(statef)
    
    # print(statef)
    # print(statef - state0)
