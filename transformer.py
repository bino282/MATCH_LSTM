import keras
import numpy as np
from keras_transformer import get_custom_objects, get_model, decode
import json
from utils import *
import pickle
import tensorflow as tf
from keras.layers import *
from keras.models import Model

def load_data(path):
    s1 = []
    s2 = []
    label= []
    with open(path,'r',encoding='utf-8') as lines:
        for line in lines:
            tokens = line.strip().split('\t')
            s1.append(preprocessor(tokens[0]))
            s2.append(preprocessor(tokens[1]))
            label.append(tokens[2])
    return s1,s2,label


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

embed_matrix = read_embed(path_embeding,embed_size=300,vocab=list(token_dict.keys()))
max_len = 100
# Generate toy data
encoder_inputs_no_padding = []
encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
for i in range(0, len(s1s_train)):
    encode_tokens, decode_tokens = s1s_train[i].split()[0:max_len-2], s2s_train[i].split()[0:max_len-2]
    encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (max_len - len(encode_tokens))
    output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (max_len - len(decode_tokens))
    decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (max_len - len(decode_tokens))
    encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
    decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
    output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
    encoder_inputs_no_padding.append(encode_tokens[:i + 2])
    encoder_inputs.append(encode_tokens)
    decoder_inputs.append(decode_tokens)
    decoder_outputs.append(output_tokens)

# Build the model
model = get_model(
    token_num = len(token_dict),
    embed_dim = 300,
    encoder_num = max_len,
    decoder_num = max_len,
    head_num = 6,
    hidden_dim = 128,
    attention_activation ='relu',
    feed_forward_activation ='relu',
    dropout_rate = 0.05,
    embed_weights = embed_matrix,
    embed_trainable = True,
)

def model_qa():
    seq1_in = model.inputs[0]
    seq2_in = model.inputs[1]
    final_rep = model.get_layer("Decoder-2-FeedForward-Norm").output(seq1_in,seq2_in)
    out = Dense(2,activation="softmax")(final_rep)
    return Model(inputs=[seq1_in,seq2_in],outputs=out)
    

model_qa = model_qa()
print(model_qa.summary())