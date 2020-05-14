# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf
import sys
import gym


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.1




class MlpModel:
    def __init__(self):
        self.sess = tf.Session(config=gpu_config)
        self.saver = tf.train.import_meta_graph('gail_models/-200.meta')
        self.saver.restore(self.sess, 'gail_models/-200') # 2641 is best
        self.graph = tf.get_default_graph()
        self.output = self.graph.get_tensor_by_name('action:0')
        self.obs = self.graph.get_tensor_by_name('Ob:0')

        self.env = gym.make('Pendulum-v0')

    def policy(self, obs):
        # 注意这里run了G，所以接下来bytes模型中保存的就是所有与G相关的图和变量
        output = self.sess.run(self.output, feed_dict={self.obs: obs.reshape([1, self.env.observation_space.shape[0]])})
        return output

    def generate_expert(self, max_steps):
        obs_list = []
        act_list = []
        obs = self.env.reset()
        sum_r = 0.
        nums_epoch = 0
        for i in range(max_steps):
            self.env.render()
            obs_list.append(obs)
            act = self.policy(obs=obs).reshape(self.env.action_space.shape[0])
            act_list.append(act)
            obs, r, d, _ = self.env.step(act)
            # self.env.render()
            sum_r += r
            if d:
                nums_epoch += 1
                obs = self.env.reset()
        return obs_list, act_list, nums_epoch, sum_r






if __name__ == '__main__':
    model = MlpModel()
    print('inference')
    obs, acts, epochs, sum_r = model.generate_expert(max_steps=20000)
    # np.savetxt('./expert_data/obs.txt', np.array(obs))
    # np.savetxt('./expert_data/act.txt', np.array(acts))
