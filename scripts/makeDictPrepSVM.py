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
import ivector_tools as it
# from guppy import hpy
# from pkgcore.config import load_config

# !/usr/bin/python -tt

# Dialect detection data preparation for SVM multi class calssifier.
# This scripts support up to six gram context
# Copyright (C) 2016, Qatar Computing Research Institute, HBKU (author: Ahmed Ali)
#
# import collections


def sortedtHashByKey(_hashFeatMap):
    _od = collections.OrderedDict(sorted(_hashFeatMap.items()))
    x = ""
    for k, v in _od.iteritems(): x += str(k) + ":" + str(v) + " "
    return x


def printHashByKey(_hashFeatMap, _ddictLen):
    x = ""
    for n in range(_ddictLen):
        if n + 1 in _hashFeatMap.keys():
            x += str(_hashFeatMap[n + 1]) + ","
        else:
            x += "0,"
    return x


def xgramList(_line, ngramCount):
    s = 0
    rlist = []
    _lineArr = _line.split()
    for idx, val in enumerate(_lineArr):
        e1 = s + 1
        e2 = s + 2
        e3 = s + 3
        e4 = s + 4
        e5 = s + 5
        e6 = s + 6
        if len(_lineArr[s:e1]) == 1: rlist.append('__'.join(_lineArr[s:e1]))
        if ngramCount >= 2 and len(_lineArr[s:e2]) == 2: rlist.append('__'.join(_lineArr[s:e2]))
        if ngramCount >= 3 and len(_lineArr[s:e3]) == 3: rlist.append('__'.join(_lineArr[s:e3]))
        if ngramCount >= 4 and len(_lineArr[s:e4]) == 4: rlist.append('__'.join(_lineArr[s:e4]))
        if ngramCount >= 5 and len(_lineArr[s:e5]) == 5: rlist.append('__'.join(_lineArr[s:e5]))
        if ngramCount >= 6 and len(_lineArr[s:e6]) == 6: rlist.append('__'.join(_lineArr[s:e6]))
        s += 1
    return rlist


# if len(sys.argv) != 7:
#     print 'usage: makeDictPrepSVM.py  all.txt dictMap.txt train.feats test.feats ngramcount features'
#     sys.exit (1)

# phoneme = sys.argv[1]
phoneme = '../all.words'
# dictFile = sys.argv[2]
dictFile = 'dict.words.1'
# trainFeat = sys.argv[3]
trainFeat = 'trainFeat_file'
devFeat = 'devFeat_file'
# testFeat = sys.argv[4]
# testFeat = 'testFeat_file'
# ngramCount = int(sys.argv[5])
ngramCount = 1
extension = 'words'

with open(phoneme) as f:
    content = f.readlines()
f.close()

langList = ['EGY', 'GLF', 'LAV', 'MSA', 'NOR']

phoneDict = {}
for line in content:
    for ngram in xgramList(line, ngramCount): phoneDict[ngram] = 1

dictFile = open(dictFile, "w")
phoneMap = {}
index = 1
for phoneGroup in phoneDict.keys():
    dictFile.write("%d %s\n" % (index, phoneGroup))
    phoneMap[phoneGroup] = index
    index += 1
dictFile.close()
_dictLength = index - 1

count = 1
trainFile = open(trainFeat, "w")

header = ""
for n in range(_dictLength):
    header += "\"P" + str(n) + "\","
header += "\"dialects\""

for index, lang in enumerate(langList):
    __file = "../data/train.vardial2017/" + lang + "." + extension
    with open(__file) as f:
        content = f.readlines()
    f.close()

    for line in content:
        id = line.split(' ', 1)[0]  # get the ID
        line = line.split(' ', 1)[1]  # remove utterance id
        featMapHash = {}
        for ngram in xgramList(line, ngramCount):
            ngram_index = phoneMap[ngram]
            if ngram_index in featMapHash.keys():
                featMapHash[ngram_index] += 1
            else:
                featMapHash[ngram_index] = 1
        x = sortedtHashByKey(featMapHash)
        trainFile.write("%s %d %s\n" % (id, index + 1, x))

        count += 1

trainFile.close()

devFile = open(devFeat, "w")
for index, lang in enumerate(langList):
    __file = "../data/dev.vardial2017/" + lang + "." + extension
    with open(__file) as f:
        content = f.readlines()
    f.close()

    for line in content:
        id = line.split(' ', 1)[0]  # get the ID
        featMapHash = {}
        for ngram in xgramList(line, ngramCount):
            try:
                ngram_index = phoneMap[ngram]
            except KeyError:
                pass
            if ngram_index in featMapHash.keys():
                featMapHash[ngram_index] += 1
            else:
                featMapHash[ngram_index] = 1
        x = sortedtHashByKey(featMapHash)
        devFile.write("%s %d %s\n" % (id, index + 1, x))
devFile.close()

# testFile = open(testFeat, "w")
# for index, lang in enumerate(langList):
#     __file = "../data/dev.vardial2017/" + lang + "." + extension
#     with open(__file) as f:
#         content = f.readlines()
#     f.close()
#
#     for line in content:
#         id = line.split(' ', 1)[0]  # get the ID
#         featMapHash = {}
#         for ngram in xgramList(line, ngramCount):
#             try:
#                 ngram_index = phoneMap[ngram]
#             except KeyError:
#                 pass
#             if ngram_index in featMapHash.keys():
#                 featMapHash[ngram_index] += 1
#             else:
#                 featMapHash[ngram_index] = 1
#         x = sortedtHashByKey(featMapHash)
#         testFile.write("%s %d %s\n" % (id, index + 1, x))
# testFile.close()

##########################################################################################################
# cat ./data/train.vardial2017/{EGY,GLF,LAV,MSA,NOR}.$features | cut -d ' ' -f 2- > all.$features.$$






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
dict_file = 'dict.words.1'
# feat_file = '../all.words'
#
#
trn_features, trn_labels, trn_names = get_feat_label(dict_file,trainFeat)
#
# feat_file = 'data/dev.vardial2017/words.c'+str(context)
dev_features, dev_labels, dev_names = get_feat_label(dict_file,devFeat)
# #
# feat_file = 'data/test.MGB3/words.c'+str(context)
# tst_features, tst_labels, tst_names = get_feat_label(dict_file, testFeat)
#
print trn_features.shape, dev_features.shape #, tst_features.shape
#
#
# trn_features, trn_labels, trn_names = get_feat_label(dict_file,feat_file)
#
# init variables
sess = tf.InteractiveSession()
siamese = siamese_model.siamese(np.shape(trn_features)[1])
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step,
                                           5000, 0.99, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss, global_step=global_step)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

saver.restore(sess, r'../suwan_model/model60400.ckpt')  # saver_folder+'/model'+str(RESTORE_STEP)+'.ckpt'

#language modeling
lang_mean=[]
for i, lang in enumerate(langList):
#     lang_mean.append(np.mean(np.append(trn_features[np.nonzero(trndev_labels == i+1)] ,dev_features[np.nonzero(dev_labels == i+1)],axis=0),axis=0))
    lang_mean.append(np.mean( trn_features[np.nonzero(trn_labels == i+1)][:],axis=0 ) )

lang_mean = np.array(lang_mean)
lang_mean = it.length_norm(lang_mean)

print np.shape(trn_features), np.shape(dev_features), np.shape(lang_mean)#,np.shape(tst_features) )

trn_features_siam = siamese.o1.eval({siamese.x1:trn_features})
# # dev_features_siam = siamese.o1.eval({siamese.x1:dev_features})
# # tst_features_siam = siamese.o1.eval({siamese.x1:tst_features})
lang_mean_siam = siamese.o1.eval({siamese.x1:lang_mean})
#
tst_scores = lang_mean_siam.dot(trn_features_siam.transpose())

hypo_lang = np.argmax(tst_scores,axis = 0)
temp = ((trn_labels-1) - hypo_lang)
acc =1- np.size(np.nonzero(temp)) / float(np.size(trn_labels))
print 'Final accurary on test dataset : %0.3f' %(acc)
# # # print(tst_scores.shape)
# # hypo_lang = np.argmax(tst_scores,axis = 0)
# # temp = ((tst_labels-1) - hypo_lang)
# # acc =1- np.size(np.nonzero(temp)) / float(np.size(tst_labels))
# # print 'Final accurary on test dataset : %0.3f' %(acc)
#
#
#
#
#
