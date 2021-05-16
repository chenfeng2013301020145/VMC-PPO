# encoding: utf-8
# compatible with 2d system

import numpy as np

class updator():
    def __init__(self, state_size):
        self.L = state_size[0]
        self.W = state_size[1]
        
        self.L3 = self.L // 2
        self.W3 = self.W // 2
        self.num = self.L3*self.W3*3
        self._Dp = state_size[-1]
        
    def state3_to_state2(self, state3):
        state2 = np.zeros([self._Dp, self.L, self.W])
        X, Y = np.meshgrid(range(self.W3), range(self.L3))
        for x,y in zip(X.reshape(-1), Y.reshape(-1)):
            state2[:, 2*y, 2*x] = state3[:, y, x, 0]
            state2[:, 2*y, 2*x + 1] = state3[:, y, x, 1]
            state2[:, 2*y + 1, 2*x] = state3[:, y, x, 2]
        return state2
        
    def state2_to_state3(self, state2):
        state3 = np.zeros([self._Dp, self.L3, self.W3, 3])
        X, Y = np.meshgrid(range(self.W3), range(self.L3))
        
        for x,y in zip(X.reshape(-1), Y.reshape(-1)):
            state3[:, y, x, :] = state2[:, 2*y:2*y+2, 2*x:2*x+2].reshape(self._Dp, 4)[:, :3]
        return state3

    def generate_mask(self, n_sample):
        rang = range(n_sample)
        swaps = np.random.randint(0, self.num, (2, n_sample))
        masks = np.arange(self.num)[None, :].repeat(n_sample, axis=0)
        masks[rang, swaps[0]], masks[rang, swaps[1]] = (masks[rang, swaps[1]], masks[rang, swaps[0]])
        # print(np.random.randint(0,10))
        return masks

    def _get_update(self, state, mask):
        state3 = self.state2_to_state3(state)
        temp = state3.reshape(self._Dp, self.num).T
        state3_f = temp[mask]
        return self.state3_to_state2(state3_f.T.reshape(state3.shape))

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from ops.HS_spin2d_Kagome import get_init_state
    
    state_size = [4,4,2]
    state0, ms = get_init_state(state_size)
    print(state0[0])
    print(state0[0].sum(0))
    test = updator(state_size)
    masks = test.generate_mask(10)
    print(masks[0])
    state_update = test._get_update(state0[0], masks[0])
    print(state_update)