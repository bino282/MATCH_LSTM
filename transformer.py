import keras
import numpy as np
from keras_transformer import get_custom_objects, get_model, decode
import json
from utils import *
import pickle
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers

def load_data(path):
    s1 = []
    s2 = []
    label= []
    with open(path,'r',encoding='utf-8') as lines:
        for line in lines:
            tokens = line.strip().split('\t')
            s1.append(preprocessor(tokens[0]))
            s2.append(preprocessor(tokens[1]))
            label.append(int(tokens[2]))
    return s1,s2,np.asarray(label)


path_embeding = '../local/word_vector/gensim_glove_vectors.txt'
s1s_train,s2s_train,labels_train= load_data('./data/train.txt')
s1s_dev,s2s_dev,labels_dev= load_data('./data/dev.txt')
s1s_test,s2s_test,labels_test= load_data('./data/test.txt')


token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}
for seq in s1s_train + s2s_train:
    for w in seq.split():
        if w not in token_dict:
            token_dict[w] = len(token_dict)

with open('voc2index.pkl','wb') as fw:
    pickle.dump(token_dict,fw)
print('vocab_size: ',len(token_dict))

max_len = 100
embed_matrix = read_embed(path_embeding,embed_size=300,vocab=list(token_dict.keys()))


def gen_toy_data(s1,s2):
    # Generate toy data
    # encoder_inputs_no_padding = []
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    for i in range(0, len(s1)):
        _s1 = [w for w in s1[i].split() if w in token_dict]
        _s2 = [w for w in s2[i].split() if w in token_dict]
        encode_tokens, decode_tokens = _s1[0:max_len-2], _s2[0:max_len-2]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (max_len-2 - len(encode_tokens))
        # output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (max_len - len(decode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (max_len-2 - len(decode_tokens))
        encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
        decode_tokens = list(map(lambda x: token_dict[x] , decode_tokens))
        # output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
        # encoder_inputs_no_padding.append(encode_tokens[:i + 2])
        encoder_inputs.append(encode_tokens)
        decoder_inputs.append(decode_tokens)
        # decoder_outputs.append(output_tokens)
    return np.asarray(encoder_inputs),np.asarray(decoder_inputs)

seq1_input,seq2_input = gen_toy_data(s1s_train,s2s_train)
seq1_input_dev,seq2_input_dev = gen_toy_data(s1s_dev,s2s_dev)
seq1_input_test,seq2_input_test = gen_toy_data(s1s_test,s2s_test)    

# Build the model
model = get_model(
    token_num = len(token_dict),
    embed_dim = 300,
    encoder_num = 3,
    decoder_num = 2,
    head_num = 6,
    hidden_dim = 256,
    attention_activation ='relu',
    feed_forward_activation ='relu',
    dropout_rate = 0.05,
    embed_weights = embed_matrix ,
    embed_trainable = True
)

def model_qa():
    seq1_in = model.inputs[0]
    seq2_in = model.inputs[1]
    decode_layer = model.get_layer("Decoder-2-FeedForward-Norm").output
    final_rep = TimeDistributed(Dense(2, use_bias=False))(decode_layer)
    final_rep = Flatten()(final_rep)
    return Model(inputs=[seq1_in,seq2_in],outputs=final_rep)
    
model_qa = model_qa()
print(model_qa.summary())
optimize = optimizers.Adam(lr=0.0001)
model_qa.compile(loss='sparse_categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])
checkpoint = ModelCheckpoint("./model_saved/model-trans-{epoch:02d}-{val_acc:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=3)


MAP_last = 0
for epoch in range(150):
    print('Train on iteration {}'.format(epoch))
    model_qa.fit([seq1_input,seq2_input],labels_train,batch_size=128,epochs=1,
                validation_data=([seq1_input_dev,seq2_input_dev],labels_dev))
    y_pred = model_qa.predict([seq1_input_dev,seq2_input_dev])
    MAP_dev,MRR_dev = map_score(s1s_dev,s2s_dev,y_pred,labels_dev)
    print('MAP_dev = {}, MRR_dev = {}'.format(MAP_dev,MRR_dev))
    if(MAP_dev>MAP_last):
        model_qa.save('./model_saved/model-trans.h5')
        print('Model saved !')
        MAP_last = MAP_dev                                              
    y_test = model_qa.predict([seq1_input_test,seq2_input_test])
    MAP_test,MRR_test = map_score(s1s_test,s2s_test,y_test,labels_test)
    print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))