import keras.activations as activations
from keras.engine.topology import Layer
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, TimeDistributed, BatchNormalization
from keras.layers.merge import concatenate, add, multiply
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda, Permute, RepeatVector
from keras.layers.recurrent import GRU, LSTM
from keras.layers import CuDNNGRU, CuDNNLSTM

def rnn_model(input_nodes, N, pfx=''):
    qi_rnn, si_rnn, nc = rnn_input(N, pfx=pfx, dropout=conf['rnn_dropout'], sdim=conf['sdim'], 
                            rnnbidi_mode=conf['rnnbidi_mode'], rnn=conf['rnn'], rnnact=conf['rnnact'], 
                            rnninit=conf['rnninit'], inputs=input_nodes, return_sequence=False)

    if conf['proj']:
        qi_rnn, si_rnn = projection_layer([qi_rnn, si_rnn], nc)

    return [qi_rnn, si_rnn]

def rnn_input(N, dropout=3/4, sdim=2, rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode=add, 
              inputs=None, return_sequence=True, pfx=''):
    if rnnbidi_mode == concatenate:
        sdim /= 2
    shared_rnn_f = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, conf['pad'], N), 
                       return_sequences=return_sequence, name='rnnf'+pfx)
    shared_rnn_b = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, conf['pad'], N),
                       return_sequences=return_sequence, go_backwards=True, name='rnnb'+pfx)
    qi_rnn_f = shared_rnn_f(inputs[0])
    si_rnn_f = shared_rnn_f(inputs[1])
    
    qi_rnn_b = shared_rnn_b(inputs[0])
    si_rnn_b = shared_rnn_b(inputs[1])
    
    qi_rnn = Activation(rnnact)(BatchNormalization()(rnnbidi_mode([qi_rnn_f, qi_rnn_b])))
    si_rnn = Activation(rnnact)(BatchNormalization()(rnnbidi_mode([si_rnn_f, si_rnn_b])))
    
    if rnnbidi_mode == concatenate:
        sdim *= 2
        
    qi_rnn = Dropout(dropout, noise_shape=(int(N*sdim),))(qi_rnn)
    si_rnn = Dropout(dropout, noise_shape=(int(N*sdim),))(si_rnn)
    
    return (qi_rnn, si_rnn, int(N*sdim))
def ap_model(input_nodes, N, pfx=''):
    if conf['w_feat_model'] == 'rnn':
        qi_feat, si_feat, adim = rnn_input(N, pfx=pfx, dropout=conf['rnn_dropout'], sdim=conf['sdim'],
                                rnnbidi_mode=conf['rnnbidi_mode'], rnn=conf['rnn'], rnnact=conf['rnnact'],
                                rnninit=conf['rnninit'], inputs=input_nodes, return_sequence=True)
                                # shapes of qi_feat and si_feat should be (batch_size, pad, adim)
    elif conf['w_feat_model'] == 'cnn':
        qi_feat, si_feat, adim = conv_aggregate(N, dropout=conf['cnn_dropout'], l2reg=conf['l2reg'], 
                                cnninit=conf['cnninit'], cnnact=conf['cnnact'], input_dim=N, inputs=input_nodes, 
                                cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}, pfx='conv_aggre_q'+pfx)
                                # shapes of qi_feat and si_feat should be (batch_size, pad, adim)
    else:
        print ('Invalid model selection')
        exit(-1)

    # Similarity measure using a bilinear form followed by a non-linear activation
    # G = tanh(QUA_T)
    G = BiLinearLayer(adim=adim, qlen=conf['pad'], alen=conf['pad'], dropout=conf['bll_dropout'], pfx=pfx)([qi_feat, si_feat]) # shape=(batch_size, pad, pad)

    # row-wise max pooling
    r_wise_max_layer = Lambda(name='r_wise_max'+pfx, function=lambda x: K.max(x, axis=2), output_shape=lambda shape:(shape[0], shape[1]))
    g_q = r_wise_max_layer(G) # shape=(batch_size, pad)
    g_q = Activation('softmax')(g_q)
    
    # column-wise max pooling
    c_wise_max_layer = Lambda(name='c_wise_max'+pfx, function=lambda x: K.max(x, axis=1), output_shape=lambda shape:(shape[0], shape[2]))
    g_a = c_wise_max_layer(G) # shape=(batch_size, pad)
    g_a = Activation('softmax')(g_a)
    
    # compute the weighted average of word features
    attn = RepeatVector(int(adim))(g_q)
    attn = Permute((2,1))(attn)
    qi_attn = multiply([qi_feat, attn]) # shape=(batch_size, pad, adim)
    avg_layer = Lambda(name='avg'+pfx, function=lambda x: K.mean(x, axis=1), output_shape=lambda shape:(shape[0],) + shape[2:])
    qi_attn = avg_layer(qi_attn) # shape=(batch_size, adim)
    
    attn = RepeatVector(int(adim))(g_a)
    attn = Permute((2,1))(attn)
    si_attn = multiply([si_feat, attn])
    si_attn = avg_layer(si_attn)
    
    if conf['proj']:
        qi_attn, si_attn = projection_layer([qi_attn, si_attn], adim) 
    
    return [qi_attn, si_attn]

def conv_aggregate(pad, dropout=1/2, l2reg=1e-4, cnninit='glorot_uniform', cnnact='relu',
        cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2, 6: 1/2, 7: 1/2}, inputs=None, input_dim=304, pfx=''):
    qi_cnn_res_list = []
    si_cnn_res_list = []
    tot_len = 0
    for fl, cd in cdim.items():
        nb_filter = int(input_dim*cd)
        shared_conv = Convolution1D(name=pfx+'conv%d'%(fl), input_shape=(None, conf['pad'], input_dim),
                    kernel_size=fl, filters=nb_filter, activation='linear', padding='same',
                    kernel_regularizer=l2(l2reg), kernel_initializer=cnninit)
        qi_one = Activation(cnnact)(BatchNormalization()(shared_conv(inputs[0]))) # shape:(None, pad, nbfilter)
        si_one = Activation(cnnact)(BatchNormalization()(shared_conv(inputs[1]))) # shape:(None, pad, nbfilter)

        qi_cnn_res_list.append(qi_one)
        si_cnn_res_list.append(si_one)

        tot_len += nb_filter
    
    qi_cnn = Dropout(dropout, noise_shape=(None, pad, tot_len))(concatenate(qi_cnn_res_list))
    si_cnn = Dropout(dropout, noise_shape=(None, pad, tot_len))(concatenate(si_cnn_res_list))
    
    return (qi_cnn, si_cnn, tot_len)

class BiLinearLayer(Layer): 
    def __init__(self, adim, qlen, alen, dropout, pfx, **kwargs): 
        self.adim = adim 
        self.qlen = qlen
        self.alen = alen
        self.dropout = dropout
        self.pfx = pfx
        super(BiLinearLayer, self).__init__(**kwargs) 
 
    def build(self, input_shape): 
        mean = 0.0 
        std = 1.0 
        # U : adim*adim 
        adim = self.adim 
        initial_U_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(adim,adim))
        self.U = K.variable(initial_U_values, name='bilinear'+self.pfx)
        self.trainable_weights = [self.U] 
        
    def call(self, inputs, mask=None): 
        if type(inputs) is not list or len(inputs) <= 1: 
            raise Exception('BiLinearLayer must be called on a list of tensors ' 
                            '(at least 2). Got: ' + str(inputs)) 
        Q = inputs[0]
        A = inputs[1]
        QU = K.dot(Q,self.U) # shape=(None, pad, adim)
        AT = Permute((2,1))(A) # shape=(None, adim, pad)
        QUA_T = K.batch_dot(QU, AT)
        QUA_T = K.tanh(QUA_T) # shape=(pad, pad)
        return QUA_T

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.qlen, self.alen)