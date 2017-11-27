# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import siamese_model_words as siamese_model



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

# todo: do we really need this file
phoneme = '../all.words'
# dictFile = 'dict.words.1'
trainFeat = 'trainFeat_file'
devFeat = 'devFeat_file'
ngramCount = 1
extension = 'words'

with open(phoneme) as f:
    content = f.readlines()
f.close()

langList = ['EGY', 'GLF', 'LAV', 'MSA', 'NOR']

phoneDict = {}
for line in content:
    for ngram in xgramList(line, ngramCount):
        phoneDict[ngram] = 1

phoneMap = {}
index = 1
for phoneGroup in phoneDict.keys():
    phoneMap[phoneGroup] = index
    index += 1

_dictLength = index - 1
input_dim = 41657
#
# init variables
sess = tf.InteractiveSession()
siamese = siamese_model.siamese(input_dim)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step,
                                           5000, 0.99, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss, global_step=global_step)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

saver.restore(sess, r'../suwan_model/model60400.ckpt')  # saver_folder+'/model'+str(RESTORE_STEP)+'.ckpt'

# utterance = "انا مقلتش كدا خالص إمبارح"
lang_mean_siam = np.load(r'../lang_mean_words.npy')
# print lang_mean_siam.shape

# builder = tf.saved_model_builder.SavedModelBuilder(r'../suwan_model/')
#
# with tf.Session(graph=tf.Graph()) as sess:
#   builder.add_meta_graph_and_variables(sess,
#                                        [tag_constants.TRAINING])
# # Add a second MetaGraphDef for inference.
# with tf.Session(graph=tf.Graph()) as sess:
#   builder.add_meta_graph([tag_constants.SERVING])
# builder.save()

utterance = "gAly EEEEEEEEEE EEEEEEEEE gAlyA jdA sssss sssss ssssss sssssss ssssss"
utt_indxes = dict()

utt_ft = np.zeros((1, input_dim), dtype='float32')

for ngram in xgramList(utterance, ngramCount):
    ngram_index = phoneMap.get(ngram, None)
    if not ngram_index:
        continue
    if ngram_index in utt_indxes.keys():
        utt_indxes[ngram_index] += 1
    else:
        utt_indxes[ngram_index] = 1
else:
    if len(utt_indxes) != 0 and len(utterance) != 0:
        for ph_idx, word_count in utt_indxes.items():
            utt_ft[0][ph_idx-1] = word_count
        utt_ft_siam = siamese.o1.eval({siamese.x1: utt_ft})
        utt_scores = lang_mean_siam.dot(utt_ft_siam.transpose())
        hypo_lang = np.argmax(utt_scores, axis=0)
        print repr(utterance)
        print langList[hypo_lang.squeeze()]
    else:
        print('unknown')
    # sorted_utt_indxes = sortedtHashByKey(utt_indxes)





