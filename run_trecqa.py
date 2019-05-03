from scipy import stats
import numpy as np
import sys,os

import utils_trecqa
import csv

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

from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
def ranknet(y_true, y_pred):
    return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1-y_true) * y_pred))), axis=-1)
def load_data_from_file(dsfile):
    #load a dataset in the csv format;
    q = [] # a set of questions
    sents = [] # a set of sentences
    labels = [] # a set of labels

    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            label = int(l['label'])
            labels.append(label)
            try:
                qtext = l['qtext'].decode('utf8')
                stext = l['atext'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['qtext']
                stext = l['atext']
            
            q.append(qtext.split(' '))
            sents.append(stext.split(' '))
            
    return (q, sents, labels)
def make_model_inputs(qi, si, f01, f10, q, sents, y):
    inp = {'qi': qi, 'si': si, 'f01':f01, 'f10':f10, 'q':q, 'sents':sents, 'y':y} 
    
    return inp
def load_set(fname, vocab=None, iseval=False):
    q, sents, y = load_data_from_file(fname)
    if not iseval:
        vocab = utils_trecqa.Vocabulary(q + sents) 
    
    pad = conf['pad']
    
    qi = vocab.vectorize(q, pad=pad)  
    si = vocab.vectorize(sents, pad=pad)        
    f01, f10 = utils_trecqa.sentence_flags(q, sents, pad)  
    
    inp = make_model_inputs(qi, si, f01, f10, q, sents, y)
    if iseval:
        return (inp, y)
    else:
        return (inp, y, vocab)   
def load_data(trainf, valf, testf):
    global vocab, inp_tr, inp_val, inp_test, y_train, y_val, y_test
    inp_tr, y_train, vocab = load_set(trainf, iseval=False)
    inp_val, y_val = load_set(valf, vocab=vocab, iseval=True)
    inp_test, y_test = load_set(testf, vocab=vocab, iseval=True)

def config():
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
    c['rnn'] = LSTM
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
    
    # Attentive Pooling-LSTM/CNN
    c['w_feat_model'] = 'rnn'
    c['bll_dropout'] = 1/2
    
    # self attention model
    c['self_pdim'] = 1/2

    # mlp scoring function
    c['Ddim'] = 2
    
    ps, h = utils_trecqa.hash_params(c)

    return c, ps, h
def embedding():
    '''
    Declare all inputs (vectorized sentences and NLP flags)
    and generate outputs representing vector sequences with dropout applied.  
    Returns the vector dimensionality.       
    '''
    pad = conf['pad']
    dropout = conf['inp_e_dropout']
    
    # story selection
    input_qi = Input(name='qi', shape=(pad,), dtype='int32')                          
    input_si = Input(name='si', shape=(pad,), dtype='int32')                 
    input_f01 = Input(name='f01', shape=(pad, utils_trecqa.flagsdim))
    input_f10 = Input(name='f10', shape=(pad, utils_trecqa.flagsdim))         

    if conf['flag']:
        input_nodes = [input_qi, input_si, input_f01, input_f10]
        N = emb.N + utils_trecqa.flagsdim
    else:
        input_nodes = [input_qi, input_si]
        N = emb.N

    shared_embedding = Embedding(name='emb', input_dim=vocab.size(), input_length=pad,
                                output_dim=emb.N, mask_zero=False,
                                weights=[vocab.embmatrix(emb)], trainable=True)
    # nlp flag
    if conf['flag']:
        emb_qi = concatenate([shared_embedding(input_qi), input_f01])
        emb_si = concatenate([shared_embedding(input_si), input_f10])
    else:
        emb_qi = shared_embedding(input_qi)
        emb_si = shared_embedding(input_si)
    
    # positional encoding
    if conf['pe']:
        if conf['pe_method'] == 'fixed':
            encoding = position_encoding_fixed(pad, N)
            pe_layer = Lambda(name='pe_fixed_layer', 
                    function=lambda x: batch_multiply(x, encoding), 
                    output_shape=lambda shape:shape)
            emb_qi = pe_layer(emb_qi)
            emb_si = pe_layer(emb_si) 
        elif conf['pe_method'] == 'learned':
            encoder = Embedding(name='pe_learnable_layer', input_dim=conf['pad'], input_length=conf['pad'],
                                            output_dim=304, mask_zero=False, trainable=True)
            pos_val = K.constant(value=np.arange(conf['pad'])) # shape=(pad,)
            pos_val = K.expand_dims(pos_val, axis=0) # shape=(1, pad)
            pos_val = K.tile(pos_val, (K.shape(input_qi)[0], 1)) # shape=(batch_size_of_x, pad)
            pos_input = Input(name='pos_input', tensor=pos_val)
            input_nodes.append(pos_input)
            encoding = encoder(pos_input)

            emb_qi = add([emb_qi, encoding])
            emb_si = add([emb_si, encoding])
    
    emb_qi = Dropout(dropout, noise_shape=(None, pad, N))(emb_qi)
    emb_si = Dropout(dropout, noise_shape=(None, pad, N))(emb_si) # shape=(None, pad, N)

    emb_outputs = [emb_qi, emb_si]
    
    return N, input_nodes, emb_outputs

def batch_multiply(x, y): 
    y = K.expand_dims(y, axis=0)
    y = K.tile(y, (K.shape(x)[0], 1, 1)) 
    return multiply([x, y]) 

def position_encoding_fixed(sentence_size, embedding_size):
    """ 
    Position Encoding described in https://arxiv.org/pdf/1503.08895.pdf
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    encodig = K.variable(value=np.transpose(encoding)) # shape=(pad, N)
    return encoding
def rnn_model(input_nodes, N, pfx=''):
    qi_rnn, si_rnn, nc = rnn_input(N, pfx=pfx, dropout=conf['rnn_dropout'], sdim=conf['sdim'], 
                            rnnbidi_mode=conf['rnnbidi_mode'], rnn=conf['rnn'], rnnact=conf['rnnact'], 
                            rnninit=conf['rnninit'], inputs=input_nodes, return_sequence=False)

    if conf['proj']:
        qi_rnn, si_rnn = projection_layer([qi_rnn, si_rnn], nc)

    return [qi_rnn, si_rnn]
def projection_layer(inputs, input_size):
    input0 = inputs[0]
    input1 = inputs[1]
    for p_i in range(conf['p_layers']):
        shared_dense = Dense(name='pdeep%d'%(p_i), output_dim=int(input_size*conf['pdim']),
                activation='linear', kernel_initializer=conf['p_init'], kernel_regularizer=l2(conf['l2reg']))
        qi_proj = Activation(conf['pact'])(BatchNormalization()(shared_dense(input0)))
        si_proj = Activation(conf['pact'])(BatchNormalization()(shared_dense(input1)))
        input0 = qi_proj
        input1 = si_proj
        input_size = int(input_size * conf['pdim'])

    dropout = conf['p_dropout']
    qi_proj = Dropout(dropout, noise_shape=(input_size,))(qi_proj)
    si_proj = Dropout(dropout, noise_shape=(input_size,))(si_proj)

    return qi_proj, si_proj
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
def mlp_ptscorer(inputs, Ddim, N, l2reg, pfx='out', oact='sigmoid', extra_inp=[]):
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

def build_model():
    # input embedding         
    N, input_nodes_emb, output_nodes_emb = embedding() 
    
    # answer sentence selection
    # avg_model / rnn_model / cnn_model / rnncnn_model / qa_lstm_cnn_model / ap_model / self_attention_model
    ptscorer_inputs = ap_model(output_nodes_emb, N, pfx='S')

    scoreS = mlp_ptscorer(ptscorer_inputs, conf['Ddim'], N,  
            conf['l2reg'], pfx='outS', oact='sigmoid')                

    output_nodes = scoreS

    model = Model(inputs=input_nodes_emb, outputs=output_nodes)
    
    model.compile(loss=ranknet, optimizer=conf['opt'])
    return model
def train_and_eval(runid):
    print('Model')
    model = build_model()
    print(model.summary())
    
    print('Training')
    fit_model(model, weightsf='weights-'+runid+'-bestval.h5')
    model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    model.load_weights('weights-'+runid+'-bestval.h5')

    print('Predict&Eval (best val epoch)')
    res = eval(model)

def fit_model(model, **kwargs):
    epochs = conf['epochs']
    callbacks = fit_callbacks(kwargs.pop('weightsf'))
    
    return model.fit(inp_tr, y=y_train, validation_data=[inp_val, y_val], batch_size=conf['batch_size'],
                     callbacks = callbacks, epochs=epochs)

def fit_callbacks(weightsf):                                  
    return [utils_trecqa.AnsSelCB(inp_val['q'], inp_val['sents'], y_val, inp_val),
            ModelCheckpoint(weightsf, save_best_only=True, monitor='mrr', mode='max'),
            EarlyStopping(monitor='mrr', mode='max', patience=conf['patience'])]
def eval(model):
    res = []
    for inp in [inp_val, inp_test]:
        if inp is None:
            res.append(None)
            continue

        pred = model.predict(inp)
        res.append(utils_trecqa.eval_QA(pred, inp['q'], inp['y'], MAP=True))
    return tuple(res)
if __name__ == "__main__":
    trainf = 'data_trecqa/train-all.csv' 
    valf = 'data_trecqa/dev.csv'
    testf = 'data_trecqa/test.csv'
    glovepath = '../local/word_vector/glove.6B.300d.txt'
    params = []
    conf, ps, h = config()
    if conf['emb'] == 'Glove': 
        print('GloVe')
        emb = utils_trecqa.GloVe(N=conf['embdim'],glovepath=glovepath)
    print('Dataset')
    load_data(trainf,valf,testf)
    runid = 'Model-%x' % (h)
    print('RunID: %s  (%s)' % (runid, ps))
    train_and_eval(runid)
