from nltk.tokenize import word_tokenize
from gensim import utils
import sys
import numpy as np
import gensim
from xml.dom import minidom
import re
from keras.preprocessing import sequence
def preprocessor(sentence):
    sentence = utils.to_unicode(sentence)
    sentence = sentence.lower()
    sentence = word_tokenize(sentence)
    sentence = " ".join(sentence)
    return sentence

def read_embed(embed_path,embed_size,vocab):
    model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(embed_path)
    embedding_matrix = np.zeros((len(vocab),embed_size))
    for i in range(len(vocab)):
        try:
            embedding_vector = model_word2vec[vocab[i]]
            embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embed_size).astype("float32")

    return embedding_matrix

DB = '<< DEBUG >>'
def debug(message):
    print (DB, message)
    sys.stdout.flush()

def constructData(dataPath, fileList,mode="TRAIN",path_dev=""):
    debug('DATA IMPORT STARTED')
    labels = []
    questions = []
    commentsL = []
    subject =[]
    if(mode=="TRAIN"):
        for xmlFile in fileList:
            debug(dataPath + xmlFile)
            doc = minidom.parse(dataPath + xmlFile)
            threads = doc.getElementsByTagName("Thread")
            for tid, thread in enumerate(threads):
                relQ = thread.getElementsByTagName('RelQuestion')[0]
                Qid = relQ.getAttribute('RELQ_ID')
                Qcat = relQ.getAttribute('RELQ_CATEGORY')
                bodyQ = relQ.getElementsByTagName('RelQBody')[0]
                body = bodyQ._get_firstChild().data if bodyQ._get_firstChild() is not None else ''
                subjQ = relQ.getElementsByTagName('RelQSubject')[0]
                subj = subjQ._get_firstChild().data if subjQ._get_firstChild() is not None else ''
                questions.append( (Qid, preprocessor(body),Qcat,preprocessor(subj)) )
                comments = []
                for relC in thread.getElementsByTagName('RelComment'):
                    Cid = relC.getAttribute('RELC_ID')
                    label = relC.getAttribute('RELC_RELEVANCE2RELQ')
                    if("multiline" in xmlFile):
                        comment = relC.getElementsByTagName('RelCClean')[0]._get_firstChild().data
                    else:
                        comment = relC.getElementsByTagName('RelCText')[0]._get_firstChild().data
                    user = relC.getAttribute('RELC_USERID')
                    comments.append( (Cid, preprocessor(comment), label,user) )
                commentsL.append(comments)
    if(mode=="DEV"):
        dict_dev ={}
        with open(path_dev,"r") as lines:
            for line in lines:
                tok = line.strip().split("\t")
                if(tok[4]=="true"):
                    dict_dev[tok[1]] = "Good"
                else:
                    dict_dev[tok[1]] = "Bad"
        for xmlFile in fileList:
            debug(dataPath + xmlFile)
            doc = minidom.parse(dataPath + xmlFile)
            threads = doc.getElementsByTagName("Thread")
            for tid, thread in enumerate(threads):
                relQ = thread.getElementsByTagName('RelQuestion')[0]
                Qid = relQ.getAttribute('RELQ_ID')
                Qcat = relQ.getAttribute('RELQ_CATEGORY')
                bodyQ = relQ.getElementsByTagName('RelQBody')[0]
                body = bodyQ._get_firstChild().data if bodyQ._get_firstChild() is not None else ''
                subjQ = relQ.getElementsByTagName('RelQSubject')[0]
                subj = subjQ._get_firstChild().data if subjQ._get_firstChild() is not None else ''
                questions.append( (Qid, preprocessor(body),Qcat,preprocessor(subj)))
                comments = []
                for relC in thread.getElementsByTagName('RelComment'):
                    Cid = relC.getAttribute('RELC_ID')
                    label = dict_dev[Cid]
                    if("multiline" in xmlFile):
                        comment = relC.getElementsByTagName('RelCClean')[0]._get_firstChild().data
                    else:
                        comment = relC.getElementsByTagName('RelCText')[0]._get_firstChild().data
                    user = relC.getAttribute('RELC_USERID')
                    comments.append( (Cid, preprocessor(comment), label,user) )
                commentsL.append(comments)
    debug('DATA IMPORT FINISHED')
    return zip(questions, commentsL)


def read_constructData(data_contruct):
    dict_cat = {'Life in Qatar':0, 'Qatar Living Tigers....':1, 'Computers and Internet':2, 'Language':3, 'Family Life in Qatar':4, 'Pets and Animals':5, 'Advice and Help':6, 'Opportunities':7, 'Cars and driving':8, 'Politics':9, 'Environment':10, 'Beauty and Style':11, 'Moving to Qatar':12, 'Qatari Culture':13, 'Salary and Allowances':14, 'Welcome to Qatar':15, 'Cars':16, 'Working in Qatar':17, 'Doha Shopping':18, 'Health and Fitness':19, 'Investment and Finance':20, 'Qatar Living Lounge':21, 'Socialising':22, 'Education':23, 'Missing home!':24, 'Sightseeing and Tourist attractions':25, 'Electronics':26, 'Qatar 2022':27, 'Visas and Permits':28, 'Funnies':29, 'Sports in Qatar':30}
    labels={"Bad":0,"PotentiallyUseful":0,"Good":1}
    s1s= []
    s2s = []
    subj = []
    y = []
    cat = []
    users = []
    for thread in data_contruct:
        s1 = thread[0][1]
        if(len(s1)==0):
            continue
        for a in thread[1]:
            s2 = a[1]
            if(len(s2)==0):
                continue
            s1s.append(s1)
            s2s.append(s2)
            subj.append(thread[0][3])
            y.append(labels[a[2]])
            users.append(a[3])
            try:
                cat.append(dict_cat[thread[0][2].strip()])
            except:
                cat.append(6)
            
    return s1s,s2s,subj,users,y,cat

def creat_vocab(data):
    vocab = set()
    voc2index = {}
    index2voc = {}
    for text in data:
        for w in text.split():
            vocab.add(w)
    vocab = list(vocab)
    vocab.insert(0, "<PAD>")
    for i in range(len(vocab)):
        voc2index[vocab[i]] = i
        index2voc[i] = vocab[i]
    return vocab,voc2index,index2voc

def convert_data_to_index(string_data, vocab):
    index_data = []
    string_data = string_data.split()
    for i in range(len(string_data)):
        if string_data[i] in vocab:
            index_data.append(vocab[string_data[i]])
    return index_data
def convertData_model(data,voc2index,max_len):
    data_model = [convert_data_to_index(x,voc2index) for x in data]
    data_model = sequence.pad_sequences(data_model,maxlen=max_len,padding='post',truncating="post")
    return data_model

def to_vector(q_w,voc2index,max_len):
    q_w_mat = convert_data_to_index(q_w,voc2index)
    q_w_mat = sequence.pad_sequences([q_w_mat],maxlen=max_len,padding='post',truncating="post")
    return q_w_mat
def predictAux(q_w, c_w,voc2index,max_len,model):
    q_v = to_vector(q_w,voc2index,max_len)
    c_v = to_vector(c_w,voc2index,max_len)
    pred = model.predict([q_v,c_v])[0]
    if pred[1] > 0.5:
        return pred[1], 'true'
    return pred[1], 'false'


def map_score(s1s_dev,s2s_dev,y_pred,labels_dev):
    QA_pairs = {}
    for i in range(len(s1s_dev)):
        pred = y_pred[i]

        s1 = " ".join(s1s_dev[i])
        s2 = " ".join(s2s_dev[i])
        if s1 in QA_pairs:
            QA_pairs[s1].append((s2, labels_dev[i], pred[1]))
        else:
            QA_pairs[s1] = [(s2, labels_dev[i], pred[1])]

    MAP, MRR = 0, 0
    num_q = len(QA_pairs.keys())
    for s1 in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

        for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)
        if(p==0):
            AP = 0
        else:
            AP /= p
        MAP += AP
    MAP /= num_q
    MRR /= num_q
    return MAP,MRR