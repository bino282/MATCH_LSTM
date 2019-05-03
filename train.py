import json
from utils import *
import pickle
import tensorflow as tf
from match_lstm import MatchLSTM

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

vocab, voc2index, index2voc = creat_vocab(s1s_train + s2s_train)

with open('voc2index.pkl','wb') as fw:
    pickle.dump(voc2index,fw)
print('vocab_size: ',len(vocab))

embed_matrix = read_embed(path_embeding,embed_size=300,vocab=vocab)

max_len_q = 150
max_len_a = 150
max_len_s = 150
seq1_input = convertData_model(s1s_train,voc2index,max_len=max_len_q)
seq2_input = convertData_model(s2s_train,voc2index,max_len=max_len_a)
labels_train = np.asarray(labels_train)


seq1_input_dev = convertData_model(s1s_dev,voc2index,max_len=max_len_q)
seq2_input_dev = convertData_model(s2s_dev,voc2index,max_len=max_len_a)
labels_dev = np.asarray(labels_dev)

seq1_input_test = convertData_model(s1s_test,voc2index,max_len = max_len_q)
seq2_input_test = convertData_model(s2s_test,voc2index,max_len = max_len_a)
labels_test = np.asarray(labels_test)

batch_size = 256
num_batch = seq1_input.shape[0]//batch_size
with tf.Session() as sess:
    model = MatchLSTM(vocab_size=len(vocab), sentence_size = max_len_q, embedding_size=300,
                          word_embedding = embed_matrix, session=sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print("=" * 50)
    print("List of Variables:")
    for v in tf.trainable_variables():
        print(v.name)
    print("=" * 50)

    _map_dev = 0
    for epoch in range(300):
        print('----------------Train on iteration {} --------------------'.format(epoch))
        for i in range(num_batch+1):
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
        if(MAP_dev > _map_dev):
            save_path = saver.save(sess, "./tmp/model-epochs-{}-{}-{}.ckpt".format(epoch,MAP_dev,MAP_test))
            print("Model saved in path: %s" % save_path)
            _map_dev = MAP_dev

    
    
