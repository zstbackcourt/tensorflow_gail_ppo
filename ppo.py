# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
from logger import MyLogger
import tensorflow as tf
from expert import Sampler
from models import Model
from runner import Runner
from collections import deque

def constfn(val):
    def f(_):
        return val

    return f

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def sample_from_buffer(buffer):
    pos = np.random.randint(low=0, high=buffer["obs"].__len__())
    tmp = []
    for key in buffer:
        tmp.append(buffer[key][pos])
    return tuple(tmp)

def buffer_clear(buffer):
    for key in buffer:
        buffer[key].clear()

def save(sess, saver, save_path, global_step):
    saver.save(sess, save_path, global_step)
    print("save models successfully!")

def load(sess, saver,save_path):
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored saved models successfully!")
    else:
        print("failed to restored saved models!")




def learn(*,
          policy,
          env,
          buffer_size,
          nminibatches,
          total_timesteps,
          ent_coef,
          lr,
          vf_coef,
          max_grad_norm,
          gamma,
          lam,
          log_interval,
          noptepochs,
          cliprange,
          save_interval,
          bc,
          bc_steps,
          algo,
          log_dir,
          save_path):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)  # 方法用来检测对象是否可被调用
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)


    total_timesteps = int(total_timesteps)
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch_train = buffer_size//nminibatches
    mylogger = MyLogger(log_dir=log_dir)
    if algo == 'gail':
        sampler = Sampler(buffer_size=buffer_size)



    sess = tf.Session()
    make_model = lambda: Model(sess=sess,
                               policy=policy,
                               ob_space=ob_space,
                               ac_space=ac_space,
                               nbatch_act=1,
                               nbatch_train=nbatch_train,
                               ent_coef=ent_coef,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)

    model = make_model()
    runner = Runner(sess=sess,env=env,model=model,buffer_size=buffer_size,gamma=gamma,lam=lam,algo=algo)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    load(sess=sess, saver=saver,save_path=save_path)
    mylogger.add_sess_graph(sess.graph)
    epinfobuf = deque(maxlen=100)

    nupdates = total_timesteps // buffer_size # 总更新轮数为最大采样步长//buffer_size

    buffer_name = ['obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'states', 'epinfos', 'ep_r','ep_count']
    buffer = dict(zip(buffer_name, [[] for _ in range(len(buffer_name))]))

    # behavior cloning
    if bc == True:
        for i in range(1, bc_steps + 1):
            print("bc iter: ", i)
            mse = 0
            for epoch in range(100):
                mb_obs, mb_act = sampler.bc_sample(batch_size=512)
                mse += model.behavior_clone(obs=mb_obs, actions=mb_act)
            print("mse", mse)
            mylogger.write_summary_scalar(iteration=i, tag="bc_mse", value=mse)
            if i % 20 == 0:
                save(sess=sess, saver=saver, save_path=save_path+"bc", global_step=i)

    # training
    for update in range(1, nupdates + 1):
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        assert buffer_size % nminibatches == 0

        inds = np.arange(buffer_size)
        np.random.shuffle(inds)
        states = runner.states
        mblossvals = []

        # 设置generator和disciminator的更新比
        critic_g = 2
        critic_d = 5
        if update % 20==0 or update<50:
            critic_g = 2
            critic_d = 20

        if states is None:
            # 非rnn policy
            for i in range(critic_g):
                inds = np.arange(buffer_size)
                # 采样一个buffer的数据
                bf_temp = obs, returns, masks, actions, values, neglogpacs, states, epinfos, ep_r, ep_count = runner.run()
                for key_idx in range(buffer_name.__len__()):
                    # 将数据保存在临时的buffer中用来更新Discriminator
                    buffer[buffer_name[key_idx]].append(bf_temp[key_idx])
                mylogger.write_summary_scalar(update, 'epr_sum', ep_r)
                mylogger.write_summary_scalar(update, 'nums of episodes', ep_count)
                mylogger.write_summary_scalar(update, 'epr_mean',ep_r//ep_count)
                epinfobuf.extend(epinfos)
                for _ in range(noptepochs):
                    # 一个buffer的数据更新noptepochs次
                    np.random.shuffle(inds)
                    for start in range(0, buffer_size,nbatch_train):
                        # 从buffer中随机采样batch
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow,*slices))
        else:
            # rnn policy
            nenvs = 100
            for i in range(4):
                bf_temp = obs, returns, masks, actions, values, neglogpacs, states, epinfos, ep_r, ep_count = runner.run()
                for key_idx in range(buffer_name.__len__()):
                    buffer[buffer_name[key_idx]].append(bf_temp[key_idx])
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(buffer_size).reshape(nenvs, -1)
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = np.array([model.initial_state] * envsperbatch).reshape([envsperbatch, -1])
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        if algo == 'ppo':
            pass
        else:
            # gail中训练Discriminator
            for i in range(critic_d):
                expert_s, expert_a = sampler.random_sample()
                gen_s, returns, masks, gen_a, values, neglogpacs, states, epinfos, ep_r, ep_count \
                    = sample_from_buffer(buffer=buffer)
                for _ in range(5):
                    runner.discriminator.train(sess, expert_s, expert_a, gen_s, gen_a)

                expert_rewards = runner.discriminator.get_rewards_e(sess, expert_s, expert_a)
                gen_rewards = runner.discriminator.get_rewards(sess, gen_s, gen_a)
                gan_loss = runner.discriminator.get_ganLoss(sess, expert_s, expert_a, gen_s, gen_a)
                mylogger.write_summary_scalar(update, 'expert_reward mean', np.mean(expert_rewards))
                mylogger.write_summary_scalar(update, 'gen_rewards mean', np.mean(gen_rewards))
                mylogger.write_summary_scalar(update, 'discrinator loss', np.mean(gan_loss))

        buffer_clear(buffer)
        lossvals = np.mean(mblossvals, axis=0)
        mblossvals.clear()
        '''pg_loss, vf_loss, entropy'''
        mylogger.write_summary_scalar(update, "pg_loss", lossvals[0])
        mylogger.write_summary_scalar(update, "vf_loss", lossvals[1])
        mylogger.write_summary_scalar(update, "entropy", lossvals[2])
        mylogger.write_summary_scalar(update, "surrogate loss", lossvals[3])
        mylogger.write_summary_scalar(update, 'critic_d', critic_d)
        mylogger.write_summary_scalar(update, 'critic_g', critic_g)
        if save_interval and (update % save_interval == 0 or update == 1):
            save(global_step=update + bc_steps, saver = saver, sess=sess, save_path=save_path)
    sess.close()
    env.close()


