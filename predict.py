import os
import sys
import json
import numpy as np
from numpy.linalg import norm
import gensim
from keras.preprocessing import sequence
from keras.models import load_model
import pickle
import scipy.stats as stats
from utils.utils import to_vector,constructData,debug
from keras import backend as K
from keras.engine.topology import Layer
from layers.attention import Position_Embedding,Attention


config = json.load(open('config.json', 'r'))
model = load_model("./model_saved/model-lstm-cnn.h5",custom_objects={'Position_Embedding':Position_Embedding,'Attention':Attention})
max_len = 100
with open('voc2index.pkl','rb') as fr:
  voc2index  = pickle.load(fr)
def predictAux(q_w, c_w,s_w,model):
    q_v = to_vector(q_w,voc2index,max_len= max_len)
    c_v = to_vector(c_w,voc2index,max_len= max_len)
    s_v = to_vector(s_w,voc2index,max_len= max_len)
    pred = model.predict([q_v,c_v])[0]
    if pred[1] > 0.5:
        return pred[1], 'true'
    return pred[1], 'false'

def predict(data, output,model):
    out = open(output, 'w')
    for q, cl in data:
        scores = []
        q_w = q[1]
        for j, c in enumerate(cl):
            c_w = c[1]
            if(len(c_w)==0):
                c_w ="<PAD>"
            s_w = q[3]
            score, pred = predictAux(q_w, c_w,s_w,model)
            scores.append( [ score, j, 0, pred ] )
        scores = sorted(scores, key=lambda score: score[0], reverse=True)
        for i in range(len(scores)):
            scores[i][2] = i + 1
        scores = sorted(scores, key=lambda score: score[1])
        for score in scores:
            out.write('\t'.join([q[0], cl[score[1]][0], str(score[2]), str(score[0]), score[3]]))
            out.write('\n')
    out.close()

debug('======= TEST MODE =======')
dataPath = config['TEST']['path']
fileList = config['TEST']['files']
data = constructData(dataPath, fileList)
output = dataPath + config['TEST']['predictions']
predict(data, output, model)
debug('======== FINISHED ========')

