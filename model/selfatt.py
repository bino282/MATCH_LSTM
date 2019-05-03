import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers.attention import Position_Embedding, Attention


class SELF_ATT():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(0.5)(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(0.5)(seq2_embed)

        lstm = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True,dropout=self.config['dropout_rate']))
        seq1_rep_rnn = lstm(seq1_embed)
        seq2_rep_rnn = lstm(seq2_embed)

        att = Attention(8, 64)

        final_rep = att([seq1_rep_rnn,seq2_rep_rnn,seq2_rep_rnn])
        final_rep = GlobalAveragePooling1D()(final_rep)
        final_rep = Dropout(0.5)(final_rep)


        output = Dense(2, activation="softmax")(final_rep)
        model = Model(inputs=[seq1, seq2], outputs=output)
        return model