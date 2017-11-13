# The script show examples to run svm classifier using both; lexical and acoustic features

# This scipt is an example showing how to train svm classifier using the bigram lexical features.

# build SVM classifier using word sequence

features=words # currently support ivec and words
context=3 # Not applicable on the ivector features as it is already hot vector
C=1000 #regularization parameter, trade-off between training error and margin

if [[ $features == "ivec" ]] ; then  
  #The ivectors features are already 400 dimnesional numberical represtation, no need to map it to hot vector.
  echo "Using the BNF iVector SVM"
  rm -f train$$ test$$
  lang=(EGY GLF LAV MSA NOR)
  for i in "${!lang[@]}"; do
    index=$(($i+1))
    lang=${lang[$i]}
    echo processing $index $lang
    cat data/train.vardial2017/$lang.ivec | sed  "s: : $index :" | awk '{printf "%s %d ",$1,$2;for (i=3;i<=NF;i++){printf "%d:%f ",i-2,$i;if(i==NF){printf "\n"}}}' >> train$$
    cat data/dev.vardial2017/$lang.ivec   | sed  "s: : $index :" | awk '{printf "%s %d ",$1,$2;for (i=3;i<=NF;i++){printf "%d:%f ",i-2,$i;if(i==NF){printf "\n"}}}' >> test$$
  done
else
   echo "Using the $features SVM"
   cat ./data/train.vardial2017/{EGY,GLF,LAV,MSA,NOR}.$features | cut -d ' ' -f 2- > all.$features.$$
  ./scripts/makeDictPrepSVM.py all.$features.$$ dict.$features.$context train$$ test$$ $context $features
fi

#exit 111

cat train$$ | cut -d ' ' -f2- > train
cat test$$ | cut -d ' ' -f2- > test
cat test$$ | cut -d ' ' -f1 > test_id

#download and compile if needed svm_multiclass: http://download.joachims.org/svm_multiclass/current/svm_multiclass.tar.gz; copy the binsries to scripts folder
./scripts/svm_multiclass_learn -c $C train model
./scripts/svm_multiclass_classify test model hypothesis

paste -d ' ' test_id hypothesis > P; mv P hypothesis
paste -d " " test_id test> P; mv P test
cat test$$ | cut -d " " -f1,2 > reference
./scripts/eval.py reference hypothesis

#tidy up 
rm -fr test$$ train$$ test_id dict.$features all.$features.$$ train test dict.$features.$context



