import json
from utils import *
import pickle
import tensorflow as tf
from match_lstm import MatchLSTM
path_dev = './data/test/SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy'
path_test= './data/test/SemEval2017-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy'
path_embeding = '../local/word_vector/gensim_glove_vectors.txt'
config = json.load(open('config.json', 'r'))
dataPath = config['TRAIN']['path']
fileList = config['TRAIN']['files']
data_train = constructData(dataPath, fileList)
dataPath = config['DEV']['path']
fileList = config['DEV']['files']
data_dev = constructData(dataPath, fileList,mode='DEV',path_dev=path_dev)
dataPath = config['TEST']['path']
fileList = config['TEST']['files']
data_test = constructData(dataPath, fileList,mode='DEV',path_dev=path_test)

s1s_train,s2s_train,subj_train,users_train,labels_train,cat_train = read_constructData(data_train)

vocab, voc2index, index2voc = creat_vocab(s1s_train+s2s_train)

with open('voc2index.pkl','wb') as fw:
    pickle.dump(voc2index,fw)
print('vocab_size: ',len(vocab))
embed_matrix = read_embed(path_embeding,embed_size=300,vocab=vocab)
max_len_q = 150
max_len_a = 150
max_len_s = 150
seq1_input = convertData_model(s1s_train,voc2index,max_len=max_len_q)
seq2_input = convertData_model(s2s_train,voc2index,max_len=max_len_a)
subj_input = convertData_model(subj_train,voc2index,max_len = max_len_s)
s1s_len_train = [len(s.split()) for s in s1s_train]
s2s_len_train = [len(s.split()) for s in s2s_train]
s1s_len_train = np.asarray(s1s_len_train)
s2s_len_train = np.asarray(s2s_len_train)
labels_train = np.asarray(labels_train)

s1s_dev,s2s_dev,subj_dev,users_dev,labels_dev,cat_dev = read_constructData(data_dev)
seq1_input_dev = convertData_model(s1s_dev,voc2index,max_len=max_len_q)
seq2_input_dev = convertData_model(s2s_dev,voc2index,max_len=max_len_a)
subj_input_dev = convertData_model(subj_dev,voc2index,max_len=max_len_s)
s1s_len_dev = [len(s.split()) for s in s1s_dev]
s2s_len_dev = [len(s.split()) for s in s2s_dev]
s1s_len_dev = np.asarray(s1s_len_dev)
s2s_len_dev = np.asarray(s2s_len_dev)
labels_dev = np.asarray(labels_dev)

s1s_test,s2s_test,subj_test,users_test,labels_test,cat_test = read_constructData(data_test)
seq1_input_test = convertData_model(s1s_test,voc2index,max_len=max_len_q)
seq2_input_test = convertData_model(s2s_test,voc2index,max_len=max_len_a)
subj_input_test = convertData_model(subj_test,voc2index,max_len=max_len_s)
s1s_len_test = [len(s.split()) for s in s1s_test]
s2s_len_test = [len(s.split()) for s in s2s_test]
s1s_len_test = np.asarray(s1s_len_test)
s2s_len_test = np.asarray(s2s_len_test)
labels_test = np.asarray(labels_test)

batch_size = 256
num_batch = seq1_input.shape[0]//batch_size
with tf.Session() as sess:
    model = MatchLSTM(vocab_size=len(vocab), sentence_size=max_len_q, embedding_size=300,
                          word_embedding=embed_matrix, session=sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print("=" * 50)
    print("List of Variables:")
    for v in tf.trainable_variables():
        print(v.name)
    print("=" * 50)

    for epoch in range(300):
        print('----------------Train on iteration {} --------------------'.format(epoch))
        for i in range(num_batch):
            if(i%5==0):
                print('running... on batch {}'.format(i))
            loss, _, step = sess.run([model.loss_op, model.train_op, model.global_step],
                                        feed_dict={model.premises: seq1_input[i*batch_size:(i+1)*batch_size], model.hypotheses: seq2_input[i*batch_size:(i+1)*batch_size],
                                                    model.labels: labels_train[i*batch_size:(i+1)*batch_size], model.lr: 0.001})

        labels_dev_pred = sess.run(model.predict_op,
                                    feed_dict={model.premises: seq1_input_dev, model.hypotheses: seq2_input_dev,
                                                model.labels: labels_dev})
        MAP_dev,MRR_dev = map_score(s1s_dev,s2s_dev,labels_dev_pred,labels_dev)
        print('MAP_dev = {}, MRR_dev = {}'.format(MAP_dev,MRR_dev))

        labels_test_pred = sess.run(model.predict_op,
                                    feed_dict={model.premises: seq1_input_test, model.hypotheses: seq2_input_test,
                                                model.labels: labels_test})
        MAP_test,MRR_test = map_score(s1s_test,s2s_test,labels_test_pred,labels_test)
        print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))
        save_path = saver.save(sess, "/tmp/model-epochs-{}.ckpt".format(epoch))
        print("Model saved in path: %s" % save_path)

    
    
