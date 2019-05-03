# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.optimizers import Adam
from layers.DynamicMaxPooling import *
from layers.BiLSTM import BiLSTM
from layers.MultiPerspectiveMatch import MultiPerspectiveMatch
from layers.SequenceMask import SequenceMask


class BiMPM():
    """implementation of Bilateral Multi-Perspective Matching
        https://arxiv.org/pdf/1702.03814.pdf
    """
    def __init__(self, config):
        self.config = config
        self.model = self.build()

    def build(self):
        query = Input(name='query', shape=(self.config['seq1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['seq2_maxlen'],))
        query_len = Input(name='query_len', shape=(1,))
        doc_len = Input(name='doc_len', shape=(1,))

        q_mask = SequenceMask(self.config['seq1_maxlen'])(query_len)
        d_mask = SequenceMask(self.config['seq1_maxlen'])(doc_len)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])
        q_embed = embedding(query)
        q_embed = Dropout(0.5)(q_embed)
        d_embed = embedding(doc)
        d_embed = Dropout(0.5)(d_embed)
        bilstm = BiLSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'])
        q_outs, q_out = bilstm(q_embed)
        d_outs, d_out = bilstm(d_embed)

        match = MultiPerspectiveMatch(self.config['channel'])
        q_match = match([d_outs, d_out, d_mask, q_outs, q_out, q_mask])
        d_match = match([q_outs, q_out, q_mask, d_outs, d_out, d_mask])

        aggre = BiLSTM(self.config['aggre_size'], dropout=self.config['dropout_rate'])
        q_outs, q_out = aggre(q_match)
        d_outs, d_out = aggre(d_match)

        flat = Concatenate(axis=1)([q_out, d_out])
        flat = Highway()(flat)

        flat_drop = Dropout(rate=self.config['dropout_rate'])(flat)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(flat_drop)

        model = Model(inputs=[query, doc, query_len, doc_len], outputs=out_)
        return model