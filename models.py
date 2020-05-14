# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import tensorflow as tf



class Model(object):
    def __init__(self, *, sess, policy, ob_space, ac_space, nbatch_act, nbatch_train,ent_coef, vf_coef, max_grad_norm):

        self.global_step_policy = tf.Variable(0, trainable=False)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, reuse=True)
        A = train_model.pdtype.sample_placeholder([None])  # action
        ADV = tf.placeholder(tf.float32, [None])  # advantage
        R = tf.placeholder(tf.float32, [None])  # return
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])  # old -logp(action)
        OLDVPRED = tf.placeholder(tf.float32, [None])  # old value prediction
        LR = tf.placeholder(tf.float32, [])  # learning rate
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)  # -logp(action)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        '''This objective can further be augmented by adding an entropy bonus to ensure suﬃcient exploration'''
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        with tf.variable_scope('model'):
            params = tf.trainable_variables()  # 图中需要训练的变量

        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads, global_step=self.global_step_policy)
        _train_a = trainer.apply_gradients(grads, global_step=self.global_step_policy)
        _train_c = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5).minimize(vf_loss)

        # behavior_cloning
        mse = tf.square(train_model.action - A) / 2
        mse = tf.reduce_mean(mse, name='action_mse')
        bc_trainer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        _train_mse = bc_trainer.minimize(mse)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs,
                      A: actions,
                      ADV: advs,
                      R: returns,
                      LR: lr,
                      CLIPRANGE: cliprange,
                      OLDNEGLOGPAC: neglogpacs,
                      OLDVPRED: values}

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            return sess.run([pg_loss, vf_loss, entropy, loss, approxkl, clipfrac, _train], td_map)[:-1]

        def behavior_clone(obs, actions, masks=None, states=None):
            td_map = {train_model.X: obs, A: actions}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run([mse, _train_mse], td_map)[0]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.behavior_clone = behavior_clone
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

