
# 
# plot ivec features with PCA,LDA and t-SNE
# TODO: use the reduced features for training

lang=(EGY GLF LAV MSA NOR)
for i in "${!lang[@]}"; do
 index=$(($i+1))
 lang=${lang[$i]}
 echo processing $index $lang
 cat ../data/train.vardial2017/$lang.ivec | cut -d ' ' -f2-  | sed "s:^:$index :" >> train$$
 cat ../data/dev.vardial2017/$lang.ivec | cut -d ' ' -f2-  | sed "s:^:$index :" >> dev$$ 
done 

for type in train dev; do
  cat ${type}$$ | cut -d ' ' -f1 > labels
  cat ${type}$$ | cut -d ' ' -f2- > feats 
  time python plot_pca_lda_tsne.py
  for dim in lda pca tsne; do
    mv $dim.png $dim.$type.png
  done 
done 

#tidy up 
rm -fr train$$ dev$$ labels feats 




