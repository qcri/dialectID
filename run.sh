# prepare the data for the three types

#This scipt is an example showing how to train svm classifier using the bigram lexical features.

# build SVM classifier using word sequence

features=words
context=2
C=1000 #regularization parameter, trade-off between training error and margin

cat ../train/{EGY,GLF,LAV,MSA,NOR}.$features | cut -d ' ' -f 2- > all.$features
./scripts/makeDictPrepSVM.py all.$features dict.$features.$context train.$features.$context test.$features.$context $context $features

cat train.$features.$context | cut -d ' ' -f2- > train
cat test.$features.$context | cut -d ' ' -f2- > test
cat test.$features.$context | cut -d ' ' -f1 > test_id

#download and compile if needed svm_multiclass: http://download.joachims.org/svm_multiclass/current/svm_multiclass.tar.gz; copy the binsries to scripts folder
./scripts/svm_multiclass_learn -c $C train model.$features.$context.$C
./scripts/svm_multiclass_classify test model.$features.$context.$C prediction.$features.$context.$C

paste -d ' ' test_id prediction.$features.$context.$C > P; mv P prediction.$features.$context.$C
paste -d " " test_id test> P; mv P test
cat test.$features.$context | cut -d " " -f1,2 > ref
./scripts/eval.py ref prediction.$features.$context.$C



