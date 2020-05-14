# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
dataDir = './expert_data/'
class Sampler():
    def __init__(self, buffer_size):
        print("sampler expert data")
        self.obs = np.loadtxt(dataDir+'obs.txt')
        self.act = np.loadtxt(dataDir+'act.txt')
        self.act = self.act.reshape((-1,1))
        self.buffer_size = buffer_size
        self.ptr = 0

    def sample(self):
        nbatches = self.obs.shape[0] // self.buffer_size
        assert nbatches > 0, "expert data not enough"
        start = self.ptr*self.buffer_size
        end = (self.ptr+1)*self.buffer_size
        self.ptr = (self.ptr +1) % nbatches
        return self.obs[start:end, :], self.act[start:end, :]

    def random_sample(self):
        nbatches = self.obs.shape[0] // self.buffer_size
        assert nbatches > 0, "expert data not enough"
        ptr = np.asscalar(np.random.randint(0, nbatches, 1))
        start = ptr * self.buffer_size
        end = (ptr + 1) * self.buffer_size
        return self.obs[start:end, :], self.act[start:end, :]

    def bc_sample(self, batch_size):
        inds = np.arange(self.obs.shape[0])
        np.random.shuffle(inds)
        return self.obs[inds[:batch_size]], self.act[inds[:batch_size]]


if __name__ == '__main__':
    """test"""
    sampler = Sampler(7000)
    obs,act = sampler.random_sample()
    print(obs.shape,act.shape)