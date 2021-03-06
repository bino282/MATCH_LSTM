# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.contrib as contrib
import tensorflow as tf
class MatchLSTM():
    def __init__(self,vocab_size,sentence_size,embedding_size,
                word_embedding,initializer=tf.truncated_normal_initializer(stddev=0.1),
                session = tf.Session(),num_class =2,name = 'MatchLSTM',initial_lr = 0.001):

        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._we = word_embedding
        self._initializer = initializer
        self._name = name
        self._num_class = num_class
        self._sess = session
        self._initial_lr = initial_lr

        self._build_inputs_and_vars()
        self._inference()
        self._initial_optimizer()

    def _build_inputs_and_vars(self):
        self.premises = tf.placeholder(shape=[None, self._sentence_size], dtype = tf.int32,
                                       name='premises')
        self.hypotheses = tf.placeholder(shape=[None, self._sentence_size], dtype = tf.int32,
                                         name='hypotheses')
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32,
                                     name='labels')
        self._batch_size = tf.shape(self.premises)[0]

        self.lr = tf.get_variable(shape=[], dtype=tf.float32, trainable=False,
                                  initializer=tf.constant_initializer(self._initial_lr), name='lr')
        self.new_lr = tf.placeholder(shape=[], dtype=tf.float32,
                                     name='new_lr')
        self.lr_update_op = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('{}_embeddings'.format(self._name)):
            self._word_embedding = tf.Variable(self._we, dtype= tf.float32)
        self._embed_pre_no = tf.nn.embedding_lookup(self._word_embedding,self.premises)
        self._embed_pre = tf.nn.dropout(self._embed_pre_no, 0.5)
        self._embed_hyp_no = tf.nn.embedding_lookup(self._word_embedding,self.hypotheses)
        self._embed_hyp = tf.nn.dropout(self._embed_hyp_no, 0.5)

    
    def _inference(self):
        with tf.variable_scope('{}_lstm_s'.format(self._name)):
            lstm_s = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            pre_length = self._length(self.premises)
            h_s, _ = tf.nn.dynamic_rnn(lstm_s, self._embed_pre, sequence_length=pre_length,
                                       dtype=tf.float32)
            self.h_s = h_s

        # with tf.variable_scope('{}_lstm_t'.format(self._name)):
            # lstm_t = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            hyp_length = self._length(self.hypotheses)
            h_t, _ = tf.nn.dynamic_rnn(lstm_s, self._embed_hyp, sequence_length=hyp_length,
                                       dtype=tf.float32)
            self.h_t = h_t
        
        self.lstm_m = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size,
                                                forget_bias=0.0)
        h_m_arr = tf.TensorArray(dtype=tf.float32, size=self._batch_size)
        i = tf.constant(0)
        c = lambda x,y : tf.less(x,self._batch_size)
        b = lambda x, y: self._match_sent(x, y)
        res = tf.while_loop(cond=c, body=b, loop_vars=(i, h_m_arr))
        self.h_m_tensor = tf.squeeze(res[-1].stack(), axis=[1])
        with tf.variable_scope('{}_fully_connect'.format(self._name)):
            w_fc = tf.get_variable(shape=[self._embedding_size, self._num_class],
                                   initializer=self._initializer, name='w_fc')
            b_fc = tf.get_variable(shape=[self._num_class],
                                   initializer=self._initializer, name='b_fc')
            self.logits = tf.matmul(self.h_m_tensor, w_fc) + b_fc
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                logits=self.logits,
                                                                name='cross_entropy')
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name='cross_entropy_sum')
        self.loss_op = tf.div(cross_entropy_sum, tf.cast(self._batch_size, dtype=tf.float32))
        self.predict_op = tf.arg_max(self.logits, dimension=1)
        self.predict_prob = tf.nn.softmax(self.logits)[:,1]

    @staticmethod
    def _length(sequence):
        mask = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(mask, axis=-1)
        return length

    def _match_sent(self,i,h_m_arr):
        h_s_i = self.h_s[i]
        h_t_i = self.h_t[i]
        length_s_i = self._length(self.premises[i])
        length_t_i = self._length(self.hypotheses[i])

        state = self.lstm_m.zero_state(batch_size =1,dtype = tf.float32)

        k = tf.constant(0)

        c = lambda a,x,y,z,s : tf.less(a,length_t_i)
        b = lambda a,x,y,z,s : self._match_attention(a,x,y,z,s)
        res = tf.while_loop(cond=c, body=b, loop_vars=(k, h_s_i, h_t_i, length_s_i, state))
        final_state_h = res[-1].h
        h_m_arr = h_m_arr.write(i, final_state_h)
        i = tf.add(i, 1)
        return i, h_m_arr



    
    def _match_attention(self,k,h_s,h_t,length_s,state):
        h_t_k = tf.reshape(h_t[k],[1,-1])
        h_s_j = tf.slice(h_s,begin=[0,0],size = [length_s,self._embedding_size])

        with tf.variable_scope('{}_attention_w'.format(self._name)):
            w_s = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_s')
            w_t = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_t')
            w_m = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_m')
            w_e = tf.get_variable(shape=[self._embedding_size, 1],
                                  initializer=self._initializer, name='w_e')
        
        last_m_h = state.h
        sum_h = tf.matmul(h_s_j, w_s) + tf.matmul(h_t_k, w_t) + tf.matmul(last_m_h, w_m)
        e_kj = tf.matmul(tf.tanh(sum_h), w_e)
        a_kj = tf.nn.softmax(e_kj)
        alpha_k = tf.matmul(a_kj, h_s_j, transpose_a=True)
        alpha_k.set_shape([1, self._embedding_size])
        m_k = tf.concat([alpha_k, h_t_k], axis=1)
        with tf.variable_scope('{}_lstm_m'.format(self._name)):
            _, new_state = self.lstm_m(inputs=m_k, state=state)
        k = tf.add(k, 1)
        return k,h_s,h_t,length_s,new_state

    def _initial_optimizer(self):
        with tf.variable_scope('{}_step'.format(self._name)):
            self.global_step = tf.get_variable(shape=[],
                                               initializer=tf.constant_initializer(0),
                                               dtype=tf.int32,
                                               name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        self.train_op = self._optimizer.minimize(self.loss_op, global_step=self.global_step)
