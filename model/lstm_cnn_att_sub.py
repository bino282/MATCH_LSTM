import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import Adam
from layers import Match,MatchTensor
from layers.SpatialGRU import *

class MATCH_LSTM_CNN():
    def __init__(self,config):
        self.config = config
        self.conf = self.config_param()
        self.model = self.build(self.conf)
    
    def build(self,conf):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        seq3 = Input(name='seq3', shape=[self.config['seq3_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])
        N = self.config['embed_size']

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(0.5, noise_shape=(None,self.config['seq1_maxlen'],self.config['embed_size']))(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(0.5, noise_shape=(None,self.config['seq2_maxlen'],self.config['embed_size']))(seq2_embed)
        seq3_embed = embedding(seq3)
        seq3_embed = Dropout(0.5, noise_shape=(None,self.config['seq3_maxlen'],self.config['embed_size']))(seq3_embed)
        share_lstm = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))
        seq1_rep_rnn = share_lstm(seq1_embed)
        seq2_rep_rnn = share_lstm(seq2_embed)
        seq3_rep_rnn = share_lstm(seq3_embed)

        subj_rep = Lambda(lambda xin: K.mean(xin, axis=1))(seq3_rep_rnn)
        subj_rep = RepeatVector(self.config['seq3_maxlen'])(subj_rep)

        seq1_rep_rnn = merge([seq1_rep_rnn,subj_rep],mode='concat')
        seq2_rep_rnn = merge([seq2_rep_rnn,subj_rep],mode='concat')


        # calculate the sentence vector on X side using Convolutional Neural Networks
        qi_aggreg, nc_cnn = self.aggregate(self.config['seq1_maxlen'], l2reg=conf['l2reg'], cnninit=conf['cnninit'], 
                                    cnnact=conf['cnnact'], input_dim=4*self.config['hidden_size'], inputs=seq1_rep_rnn, 
                                    cdim={2: 1/2, 3: 1/2, 4: 1/2, 5:1/2}, pfx='aggre_q')
        # re-embed X,Y in attention space
        adim = int(N*conf['adim'])
        shared_dense_q = Dense(adim, kernel_regularizer=l2(conf['l2reg']), 
                            kernel_initializer=conf['p_init'])
        shared_dense_s = Dense(adim, kernel_regularizer=l2(conf['l2reg']), 
                           kernel_initializer=conf['p_init'])
        qi_aggreg_attn = BatchNormalization()(shared_dense_q(qi_aggreg))
        si_attn = TimeDistributed(shared_dense_s)(seq2_rep_rnn)
        si_attn = TimeDistributed(BatchNormalization())(si_attn)

        # apply an attention function on Y side by producing an vector of scalars denoting the attention for each token
        si_foc = self.focus(N,self.config['seq1_maxlen'],qi_aggreg_attn, si_attn, seq2_rep_rnn, conf['sdim'], adim, conf['l2reg'])

        si_aggreg, nc_cnn = self.aggregate(self.config['seq1_maxlen'], l2reg=conf['l2reg'], cnninit=conf['cnninit'], 
                                    cnnact=conf['cnnact'], input_dim=4*self.config['hidden_size'], inputs=si_foc, 
                                    cdim={2: 1/2, 3: 1/2, 4: 1/2,5:1/2})
        if conf['proj']:
            qi_aggreg, si_aggreg = self.projection_layer([qi_aggreg, si_aggreg],conf,nc_cnn)
        scoreS = self.mlp_ptscorer([qi_aggreg,si_aggreg], conf['Ddim'], N,  
                conf['l2reg'], pfx='outS', oact='softmax')                

        output_nodes = scoreS

        model = Model(inputs=[seq1,seq2,seq3], outputs=output_nodes)
        

       
        return model

    def aggregate(self,seq_len, dropout=1/2, l2reg=1e-4, cnninit='glorot_uniform', cnnact='relu',
        cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}, inputs=None, input_dim=304, pfx=''):
        cnn_res_list = []
        tot_len = 0
        for fl, cd in cdim.items():
            nb_filter = int(input_dim*cd)
            shared_conv = Convolution1D( input_shape=(None, seq_len, input_dim),
                        kernel_size=fl, filters=nb_filter, activation='linear',
                        kernel_regularizer=l2(l2reg), kernel_initializer=cnninit)
            cnn_res = Activation(cnnact)(BatchNormalization()(shared_conv(inputs)))

            pool = MaxPooling1D(pool_size=int(seq_len-fl+1))
            cnn_res = pool(cnn_res)
            cnn_res = Flatten()(cnn_res)

            cnn_res_list.append(cnn_res)

            tot_len += nb_filter

        aggreg = Dropout(dropout, noise_shape=(tot_len,))(concatenate(cnn_res_list))

        return (aggreg, tot_len)
    def focus(self,N,seq_len, input_aggreg, input_seq, orig_seq, sdim, awidth, l2reg, pfx=''):
        repeat_vec = RepeatVector(seq_len)
        input_aggreg_rep = repeat_vec(input_aggreg)

        attn = Activation('tanh')(add([input_aggreg_rep, input_seq]))

        shared_dense = Dense(1, kernel_regularizer=l2(l2reg))
        attn = TimeDistributed(shared_dense)(attn)
        attn = Flatten()(attn)

        attn = Activation('softmax')(attn)
        attn = RepeatVector(4*self.config['hidden_size'])(attn)
        attn = Permute((2,1))(attn)
        output = multiply([orig_seq, attn])

        return output

    def mlp_ptscorer(self,inputs, Ddim, N, l2reg, pfx='out', oact='sigmoid', extra_inp=[]):
        """ Element-wise features from the pair fed to an MLP. """

        sum_vec = add(inputs)
        mul_vec = multiply(inputs)

        mlp_input = concatenate([sum_vec, mul_vec])

        # Ddim may be either 0 (no hidden layer), scalar (single hidden layer) or
        # list (multiple hidden layers)
        if Ddim == 0:
            Ddim = []
        elif not isinstance(Ddim, list):
            Ddim = [Ddim]
        if Ddim:
            for i, D in enumerate(Ddim):
                shared_dense = Dense(int(N*D), kernel_regularizer=l2(l2reg), 
                                    activation='linear', name=pfx+'hdn%d'%(i))
                mlp_input = Activation('tanh')(shared_dense(mlp_input))

        shared_dense = Dense(1, kernel_regularizer=l2(l2reg), activation=oact, name=pfx+'mlp')
        mlp_out = shared_dense(mlp_input)
        
        return mlp_out
    def projection_layer(self,inputs,conf,input_size):
        input0 = inputs[0]
        input1 = inputs[1]
        for p_i in range(conf['p_layers']):
            shared_dense = Dense(int(input_size*conf['pdim']),activation='linear', kernel_initializer=conf['p_init'], kernel_regularizer=l2(conf['l2reg']))
            qi_proj = Activation(conf['pact'])(BatchNormalization()(shared_dense(input0)))
            si_proj = Activation(conf['pact'])(BatchNormalization()(shared_dense(input1)))
            input0 = qi_proj
            input1 = si_proj
            input_size = int(input_size * conf['pdim'])

        dropout = conf['p_dropout']
        qi_proj = Dropout(dropout, noise_shape=(input_size,))(qi_proj)
        si_proj = Dropout(dropout, noise_shape=(input_size,))(si_proj)

        return qi_proj, si_proj

    def config_param(self):
        c = dict()
        # embedding params
        c['emb'] = 'Glove'
        c['embdim'] = 300
        c['inp_e_dropout'] = 1/2
        c['flag'] = True
        c['pe'] = True
        c['pe_method'] = 'learned' # 'fixed' or 'learned'

        # training hyperparams
        c['opt'] = 'adadelta'
        c['batch_size'] = 160   
        c['epochs'] = 160
        c['patience'] = 155
        
        # sentences with word lengths below the 'pad' will be padded with 0.
        c['pad'] = 60
        
        # rnn model       
        c['rnn_dropout'] = 1/2     
        c['l2reg'] = 1e-4
                                                
        c['rnnbidi'] = True                      
        c['rnn'] = CuDNNLSTM
        c['rnnbidi_mode'] = concatenate
        c['rnnact'] = 'tanh'
        c['rnninit'] = 'glorot_uniform'                      
        c['sdim'] = 5

        # cnn model
        c['cnn_dropout'] = 1/2     
        c['pool_layer'] = MaxPooling1D
        c['cnnact'] = 'relu'
        c['cnninit'] = 'glorot_uniform'
        c['pact'] = 'tanh'

        # projection layer
        c['proj'] = True
        c['pdim'] = 1/2
        c['p_layers'] = 1
        c['p_dropout'] = 1/2
        c['p_init'] = 'glorot_uniform'
        
        # QA-LSTM/CNN+attention
        c['adim'] = 1/2
        c['cfiltlen'] = 3
        # mlp scoring function
        c['Ddim'] = 2

        return c