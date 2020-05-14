# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import discriminator as discriminator



def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class Runner(object):
    def __init__(self, *, sess, env, model, buffer_size, gamma, lam,algo):
        self.sess = sess
        self.env = env
        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.model = model
        self.algo = algo
        if self.algo == 'gail':
            self.discriminator = discriminator.WganDiscriminator(env)

        self.gamma = gamma
        self.lam = lam
        self.nsteps = buffer_size
        self.states = model.initial_state

    # collect generate experience
    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        epinfos = []
        ep_r = 0
        self.obs = self.env.reset()
        # 只能采样一条轨迹, 严格来说
        epi_count = 0
        last_values = 0.
        while True:
            # self.env.render()
            act, v, self.states, neglogp = self.model.step(self.obs.reshape(-1, self.obs_space.shape[0]), self.states)
            mb_obs.append(self.obs.reshape(self.obs_space.shape[0]))
            act = act.reshape(self.action_space.shape[0])
            mb_actions.append(act)
            mb_values.append(np.asscalar(v))
            mb_neglogpacs.append(neglogp[0])
            self.obs, r, d, _ = self.env.step(act)
            ep_r += r
            mb_dones.append(d)
            if self.algo == 'ppo':
                mb_rewards.append(r / 8 + 1)  # reward normalize
            else:
                r = self.discriminator.get_rewards(self.sess, self.obs.reshape(1, self.obs_space.shape[0]),
                                                   act.reshape(1, self.action_space.shape[0]))
                mb_rewards.append(r)
            if d:
                epi_count += 1
                self.obs = self.env.reset()

            if len(mb_dones) >= self.nsteps:
                last_values = self.model.value(self.obs.reshape(-1, self.obs_space.shape[0]), self.states)
                break

        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, self.obs_space.shape[0])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(self.nsteps)
        mb_rewards -= np.mean(mb_rewards)
        mb_rewards /= np.max(np.abs(mb_rewards))

        mb_actions = np.asarray(mb_actions, np.float32).reshape(self.nsteps, self.action_space.shape[0])
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape(self.nsteps)

        print(mb_obs.shape, mb_rewards.shape, mb_actions.shape, mb_values.shape, mb_neglogpacs.shape, mb_dones.shape)
        print('epi_count | ', epi_count)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0.

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns[:] = mb_advs + mb_values
        print('sum_reward | ', ep_r)
        return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                self.states, epinfos, ep_r, epi_count)