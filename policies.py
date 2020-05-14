# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf
from baselines_utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from distributions import make_pdtype


def mlp(unscaled_vector):
    h = tf.nn.relu(fc(unscaled_vector, 'fc1', nh=100, init_scale=np.sqrt(1.0)))
    h2 = tf.nn.tanh(fc(h, 'fc2', nh=100, init_scale=np.sqrt(1.0)))
    h3 = fc(h, 'fc2', nh=64, init_scale=np.sqrt(1.0))
    return h3
    # h3 = activ(fc(unscaled_vector, 'fc3', nh=256, init_scale=np.sqrt(0.2)))


def lkrelu(x, slope=0.05):
    return tf.maximum(slope * x, x)


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    # cast(x, dtype, name=None) 将x的数据格式转化成dtype.
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        # nh, nw, nc = ob_space.shape  # (nh, nw, nc) = (height, width, channels)
        ob_shape = (nbatch, ob_space.shape[0])
        # nact = ac_space.n
        # X = tf.placeholder(tf.uint8, ob_shape)  # obs
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='phOb')
        M = tf.placeholder(tf.float32, [nbatch], name='phMaskDone')  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2], name='phCellState')  # states and output: (c, h)
        with tf.variable_scope("model", reuse=reuse):
            # h = nature_cnn(X)
            # h = tf.add(X, 0, name='h')  # need more network to power enough
            h = mlp(X)
            xs = batch_to_seq(h, nenv, nsteps)  # A List contain tensors all with shape [nenv, -1]
            ms = batch_to_seq(M, nenv, nsteps)  # A List contain tensors all with shape [nenv, 1]
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            # pi = fc(h5, 'fc_pi', actdim)
            pi0 = tf.nn.tanh(h5[:, :1])*3
            pi1 = tf.nn.sigmoid(h5[:, 1:2])*10
            pi = tf.concat([pi0, pi1], axis=1, name='pi')

            vf = fc(h5, 'v', 1)
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        # self.pdtype = make_pdtype(ac_space)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        action = tf.add(a0, 0, name='action')  # use this tensor as action when inference
        newState = tf.add(snew, 0, name='newCellState')
        print('sel.pd.shape', self.pd.shape, a0.shape)
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, reuse=False, training=True):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape  # 增加nbatch行
        actdim = ac_space.shape[0]
        # X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        X = tf.placeholder(tf.float32, [None, ob_space.shape[0]], name='Ob')
        with tf.variable_scope("model", reuse=reuse):
            # activ = tf.tanh
            bn = tf.layers.batch_normalization
            activ = lkrelu
            # h1 = activ(bn(fc(X, 'pi_fc1', nh=512, init_scale=np.sqrt(2)), training=training))
            # h2 = activ(bn(fc(h1, 'pi_fc2', nh=512, init_scale=np.sqrt(2)), training=training))
            # h3 = activ(bn(fc(h2, 'pi_fc3', nh=256, init_scale=np.sqrt(2)), training=training))
            h1 = activ(fc(X, 'pi_fc1', nh=100, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=100, init_scale=np.sqrt(2)))
            # pi0 = tf.nn.tanh(fc(h2, 'pi0', 1, init_scale=0.01))*3  # (-3, 3)
            # pi1 = tf.nn.sigmoid(fc(h2, 'pi1', 1, init_scale=0.01))*10  # (0, 10)
            # pi = tf.concat([pi0, pi1], axis=1, name='pi')
            pi = tf.nn.tanh(fc(h2, 'pi', nh=actdim))*10
            # h1 = activ(bn(fc(X, 'vf_fc1', nh=512, init_scale=np.sqrt(2)), training=training))
            # h2 = activ(bn(fc(h1, 'vf_fc2', nh=512, init_scale=np.sqrt(2)), training=training))
            # h3 = activ(bn(fc(h2, 'vf_fc3', nh=256, init_scale=np.sqrt(2)), training=training))
            h1 = activ(fc(X, 'vf_fc1', nh=100, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=100, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())
            # logstd = tf.layers.dense(inputs=h2, activation=None, units=actdim, name='logstd')

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)  # pi * 0.0 + logstd的作用是使得qi有相同的形状
        self.pdtype = make_pdtype(ac_space)  # Probability distribution function  pd
        '''返回DiagGaussianPd的类'''
        self.pd = self.pdtype.pdfromflat(pdparam)
        a0 = self.pd.sample()
        self.action = tf.identity(a0, name='action')  # use this tensor as action when inference
        # if I need action clipping?
        # a1 = tf.clip_by_value(a0[:, 0:1], -3, 3)
        # a2 = tf.clip_by_value(a0[:, 1:2], 0, 10)
        # a0 = tf.concat([a1, a2], axis=1)
        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            # {X: ob}给placeholder赋值
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})
        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
