import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam

class QA_LSTM_ATT():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq2_embed = embedding(seq2)

        seq1_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(seq1_embed)
        seq1_rep = Dropout(0.3)(seq1_rep)
        seq2_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(seq2_embed)
        seq2_rep = Dropout(0.3)(seq2_rep)

        # attention model
        attn = merge([seq1_rep, seq2_rep], mode="dot", dot_axes=[1, 1])
        attn = Flatten()(attn)
        attn = Dense(self.config['seq1_maxlen'] * 2 * self.config['hidden_size'])(attn)
        attn = Reshape((self.config['seq1_maxlen'], 2* self.config['hidden_size']))(attn)

        qenc_attn = merge([seq1_rep, attn], mode="sum")
        qenc_attn = Flatten()(qenc_attn)

        output = Dense(2, activation="softmax")(qenc_attn)
        model = Model(inputs=[seq1, seq2], outputs=output)
        return model