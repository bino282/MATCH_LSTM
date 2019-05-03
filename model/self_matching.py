import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers.attention import Position_Embedding, Attention
from layers import SharedWeight,QuestionAttnGRU,Question_Pooling


class SELF_MATCH():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):

        H = self.config['hidden_size']
        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')
        VQ_r = SharedWeight(size=(H, H), name='VQ_r')

        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r]

        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(0.5)(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(0.5)(seq2_embed)


        lstm = Bidirectional(GRU(self.config['hidden_size'], return_sequences=True,dropout=self.config['dropout_rate']))
        seq1_rep_rnn = lstm(seq1_embed)
        seq2_rep_rnn = lstm(seq2_embed)

        vP = QuestionAttnGRU(units=H,
                             return_sequences=True,
                             unroll=False) ([
                                 seq2_rep_rnn, seq1_rep_rnn,
                                 WQ_u, WP_v, WP_u, v, W_g1
                             ])

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True,
                                       unroll=False)) ([
                                          vP, vP,
                                          WP_v, WPP_v, v, W_g2
                                      ])
        gP = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               unroll=False)) (hP)

        rQ = QuestionPooling() ([seq1_rep_rnn, WQ_u, WQ_v, v, VQ_r])
        rQ = Dropout(rate=self.config['dropout_rate'], name='rQ') (rQ)

       
        att = Attention(8, 64)

        final_rep = att([seq1_rep_rnn,seq2_rep_rnn,seq2_rep_rnn])
        final_rep = GlobalAveragePooling1D()(final_rep)
        final_rep = Dropout(0.5)(final_rep)


        output = Dense(2, activation="softmax")(final_rep)
        model = Model(inputs=[seq1, seq2], outputs=output)
        return model