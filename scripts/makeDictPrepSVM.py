#!/usr/bin/python -tt

# Dialect detection data preparation for SVM multi class calssifier.
# This scripts support up to six gram context
# Copyright (C) 2016, Qatar Computing Research Institute, HBKU (author: Ahmed Ali)
#

import os
import codecs
import numpy as np
import tensorflow as tf
import collections
import siamese_model_words as siamese_model
from guppy import hpy
from pkgcore.config import load_config


def sortedtHashByKey(_hashFeatMap):
    _od = collections.OrderedDict(sorted(_hashFeatMap.items()))
    x=""
    for k, v in _od.iteritems(): x+=str(k)+":"+str(v)+" "
    return x

def printHashByKey(_hashFeatMap,_ddictLen):
    x=""
    for n in range(_ddictLen):
        if n+1 in _hashFeatMap.keys(): x+=str(_hashFeatMap[n+1])+","
        else : x+="0,"
    return x

def xgramList (_line,ngramCount) :
  s=0
  rlist=[]
  _lineArr=_line.split( )
  for idx, val in enumerate(_lineArr):
    e1 = s +1
    e2 = s +2
    e3 = s +3
    e4 = s +4
    e5 = s +5
    e6 = s +6
    if len (_lineArr[s:e1]) == 1: rlist.append('__'.join(_lineArr[s:e1]))
    if ngramCount >= 2 and len (_lineArr[s:e2]) == 2: rlist.append('__'.join(_lineArr[s:e2]))
    if ngramCount >= 3 and len (_lineArr[s:e3]) == 3: rlist.append('__'.join(_lineArr[s:e3]))
    if ngramCount >= 4 and len (_lineArr[s:e4]) == 4: rlist.append('__'.join(_lineArr[s:e4]))
    if ngramCount >= 5 and len (_lineArr[s:e5]) == 5: rlist.append('__'.join(_lineArr[s:e5]))
    if ngramCount >= 6 and len (_lineArr[s:e6]) == 6: rlist.append('__'.join(_lineArr[s:e6]))
    s+=1
  return rlist



# cat ./data/train.vardial2017/{EGY,GLF,LAV,MSA,NOR}.$features | cut -d ' ' -f 2- > all.$features.$$

data_path = r'../data/train.vardial2017'
word_features = [r'EGY.words', r'GLF.words',r'LAV.words', r'MSA.words', r'NOR.words']

# if len(sys.argv) != 7:
#     print 'usage: makeDictPrepSVM.py  all.txt dictMap.txt train.feats test.feats ngramcount features'
#     sys.exit (1)
    
#phoneme = sys.argv[1]
ngramCount = 1
dictFile_path = 'dict.words.'+str(ngramCount)
# trainFeat = sys.argv[3]
# testFeat = sys.argv[4]
# extension = sys.argv[6]

content = list()

for dial_path in word_features:
    ffp = os.path.join(data_path, dial_path)
    with codecs.open(ffp, encoding='utf-8') as dial_file:
        content.extend(dial_file.readlines())


# with open(phoneme) as f:
#     content = f.readlines()
# f.close()

langList=['EGY', 'GLF', 'LAV', 'MSA','NOR']


phoneDict={}
for line in content:
    for ngram in xgramList (line,ngramCount): phoneDict [ngram] = 1

#dictFile = open(dictFile, "w")
with codecs.open(dictFile_path, encoding='utf-8', mode='w') as df:
    phoneMap={}
    index=1
    for phoneGroup in phoneDict.keys():
        df.write("%d %s\n" % (index, phoneGroup))
        phoneMap [phoneGroup] = index
        index += 1

_dictLength=index-1




# tf.reset_default_graph()
# # Create some variables.
# # v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
# # v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)
#



def get_dataset_size(dict_file,feat_file):
# Counting feature dimension and total number of utterances
    f = open(dict_file)
    dict_dim = 0
    for line in f:
        dict_dim+=1
    f.close()
    feat_len = 0
    f = open(feat_file)
    for line in f:
        feat_len+=1
    f.close()
    return dict_dim, feat_len

def get_feat_label(dict_file, feat_file):
    # Get feature vectors from files
    dict_dim, feat_len = get_dataset_size(dict_file, feat_file)
    features = np.zeros((feat_len, dict_dim), dtype='float32')
    labels = np.zeros((feat_len), dtype='int8')
    names = []
    f = open(feat_file)
    count = 0
    for line in f:
        names.append(line.split()[0])
        labels[count] = line.split()[1]
        line = line.split()[2:]
        for iter in range(0, len(line)):
            elements = line[iter].split(':')
            features[count][int(elements[0]) - 1] = elements[1]
        count = count + 1
    f.close()

    return features, labels, names


context = ngramCount
# dict_file = 'data/train.vardial2017/dict.words.c'+str(context)
# feat_file = 'data/train.vardial2017/words.c'+str(context)
dict_file = '../dict.words.1'
feat_file = '../all.words'


trn_features, trn_labels, trn_names = get_feat_label(dict_file,feat_file)

feat_file = 'data/dev.vardial2017/words.c'+str(context)
dev_features, dev_labels, dev_names = get_feat_label(dict_file,feat_file)

feat_file = 'data/test.MGB3/words.c'+str(context)
tst_features, tst_labels, tst_names = get_feat_label(dict_file,feat_file)

print trn_features.shape, dev_features.shape, tst_features.shape


trn_features, trn_labels, trn_names = get_feat_label(dict_file,feat_file)

# init variables
sess = tf.InteractiveSession()
siamese = siamese_model.siamese(np.shape(trn_features)[1])
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step,
                                           5000, 0.99, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss, global_step=global_step)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, r'../suwan_model/model60400.ckpt')  # saver_folder+'/model'+str(RESTORE_STEP)+'.ckpt'

## trn_features_siam = siamese.o1.eval({siamese.x1:trn_features})
# dev_features_siam = siamese.o1.eval({siamese.x1:dev_features})
# tst_features_siam = siamese.o1.eval({siamese.x1:tst_features})
# lang_mean_siam = siamese.o1.eval({siamese.x1:lang_mean})

# tst_scores = lang_mean_siam.dot(tst_features_siam.transpose() )
# # print(tst_scores.shape)
# hypo_lang = np.argmax(tst_scores,axis = 0)
# temp = ((tst_labels-1) - hypo_lang)
# acc =1- np.size(np.nonzero(temp)) / float(np.size(tst_labels))
# print 'Final accurary on test dataset : %0.3f' %(acc)


  
      
    
