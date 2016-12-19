#!/usr/bin/python -tt

# Dialect detection data preparation for SVM multi class calssifier.
# This scripts support up to six gram context
# Copyright (C) 2016, Qatar Computing Research Institute, HBKU (author: Ahmed Ali)
#

import sys  
reload(sys)
import codecs
import bs4 as bs
import collections
import re
from subprocess import call
sys.setdefaultencoding('utf8')

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


if len(sys.argv) != 7:
    print 'usage: makeDictPrepSVM.py  all.txt dictMap.txt train.feats test.feats ngramcount features'
    sys.exit (1)
    
phoneme = sys.argv[1]
dictFile = sys.argv[2]
trainFeat = sys.argv[3]
testFeat = sys.argv[4]
ngramCount = int(sys.argv[5])
extension = sys.argv[6]



with open(phoneme) as f:
    content = f.readlines()
f.close()

langList=['EGY', 'GLF', 'LAV', 'MSA','NOR']


phoneDict={}
for line in content:
    for ngram in xgramList (line,ngramCount): phoneDict [ngram] = 1

dictFile = open(dictFile, "w")
phoneMap={}
index=1
for phoneGroup in phoneDict.keys():
    dictFile.write("%d %s\n" % (index, phoneGroup))
    phoneMap [phoneGroup] = index
    index+=1
dictFile.close()
_dictLength=index-1

count=1
trainFile = open(trainFeat, "w")



header=""
for n in range(_dictLength):
    header+="\"P"+str(n)+"\","
header+="\"dialects\""



for index, lang in enumerate(langList):
  __file="./data/train/"+lang+"."+extension
  with open(__file) as f:
      content = f.readlines()
  f.close()
  
  for line in content:
    id=line.split(' ',1)[0] #get the ID
    line=line.split(' ',1)[1] # remove utterance id
    featMapHash = {}
    for ngram in xgramList (line,ngramCount):
      ngram_index=phoneMap[ngram]
      if ngram_index in featMapHash.keys(): featMapHash[ngram_index]+=1
      else : featMapHash[ngram_index]=1                    
    x=sortedtHashByKey (featMapHash)
    trainFile.write("%s %d %s\n" % (id,index+1, x))
    
    count+=1
    
trainFile.close()

testFile = open(testFeat, "w")
for index, lang in enumerate(langList):
  __file="./data/test/"+lang+"."+extension
  with open(__file) as f:
      content = f.readlines()
  f.close()
  
  for line in content:
    id=line.split(' ',1)[0] #get the ID
    featMapHash = {}
    for ngram in xgramList (line,ngramCount):
      try: ngram_index=phoneMap[ngram]
      except KeyError : pass
      if ngram_index in featMapHash.keys(): featMapHash[ngram_index]+=1
      else : featMapHash[ngram_index]=1                    
    x=sortedtHashByKey (featMapHash)
    testFile.write("%s %d %s\n" % (id,index+1, x))
testFile.close()
  
      
    
