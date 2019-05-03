import json
from utils import *
from model import biMPM,lstm_cnn,lstm_cnn_att_sub,selfatt,mvrnn
from keras import optimizers
import keras.backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pickle
from keras.models import load_model
from layers.attention import Position_Embedding,Attention

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

vocab, voc2index, index2voc = creat_vocab(s1s_train+s2s_train)

with open('voc2index.pkl','wb') as fw:
    pickle.dump(voc2index,fw)
print('vocab_size: ',len(vocab))
embed_matrix = read_embed(path_embeding,embed_size=300,vocab=vocab)
max_len_q = 100
max_len_a = 100
max_len_s = 100
seq1_input = convertData_model(s1s_train,voc2index,max_len=max_len_q)
seq2_input = convertData_model(s2s_train,voc2index,max_len=max_len_a)
labels_train = np.asarray(labels_train)

seq1_input_dev = convertData_model(s1s_dev,voc2index,max_len=max_len_q)
seq2_input_dev = convertData_model(s2s_dev,voc2index,max_len=max_len_a)
labels_dev = np.asarray(labels_dev)

seq1_input_test = convertData_model(s1s_test,voc2index,max_len=max_len_q)
seq2_input_test = convertData_model(s2s_test,voc2index,max_len=max_len_a)
labels_test = np.asarray(labels_test)

print(seq1_input.shape)
model_config={'seq1_maxlen':max_len_q,'seq2_maxlen':max_len_a,'seq3_maxlen':max_len_s,
                'vocab_size':len(voc2index),'embed_size':300,
                'hidden_size':300,'dropout_rate':0.5,
                'embed':embed_matrix,
                'embed_trainable':True,
                'channel':5,
                'aggre_size':100,
                'target_mode':'ranking'}
def ranknet(y_true, y_pred):
    return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1-y_true) * y_pred))), axis=-1)

try:
    model_lstm = load_model("./model_saved/model-lstm-cnn.h5")
    print("Load model success......")
except:
    print("Creating new model......")
    model_lstm = selfatt.SELF_ATT(config=model_config).model
print(model_lstm.summary())
optimize = optimizers.Adam(lr=0.0001)
model_lstm.compile(loss='sparse_categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])
checkpoint = ModelCheckpoint("./model_saved/model-lstm-cnn-{epoch:02d}-{val_acc:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=3)


MAP_last = 0
for epoch in range(150):
    print('Train on iteration {}'.format(epoch))
    model_lstm.fit([seq1_input,seq2_input],labels_train,batch_size=128,epochs=1,
                validation_data=([seq1_input_dev,seq2_input_dev],labels_dev))
    y_pred = model_lstm.predict([seq1_input_dev,seq2_input_dev])
    MAP_dev,MRR_dev = map_score(s1s_dev,s2s_dev,y_pred,labels_dev)
    print('MAP_dev = {}, MRR_dev = {}'.format(MAP_dev,MRR_dev))
    if(MAP_dev>MAP_last):
        model_lstm.save('./model_saved/model-lstm-cnn.h5')
        print('Model saved !')
        MAP_last = MAP_dev                                              
    y_test = model_lstm.predict([seq1_input_test,seq2_input_test])
    MAP_test,MRR_test = map_score(s1s_test,s2s_test,y_test,labels_test)
    print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))