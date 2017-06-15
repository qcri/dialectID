from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import keras
from keras.utils import to_categorical
seed = 7
np.random.seed(seed)

#load data
languages = ['EGY','GLF','LAV','MSA','NOR']
trn_labels = []
trn_names = []
trn_ivectors = np.empty((0,400))
dev_labels = []
dev_names = []
dev_ivectors = np.empty((0,400))
lang_mean_ivectors = np.empty((5,200))

for i,lang in enumerate(languages):
    print "Loading:", i, lang
    
    filename = '../data/train.vardial2017/%s.ivec' % lang
    name   = np.loadtxt(filename,usecols=[0],dtype='string')
    ivector = np.loadtxt(filename,usecols=range(1,401),dtype='float32')
    a = np.empty(len(ivector));a.fill(i)
    
    trn_names=np.append(trn_names,name)
    trn_ivectors = np.append(trn_ivectors, ivector,axis=0)
    trn_labels = np.append(trn_labels, a, axis=0)
    
    filename = '../data/dev.vardial2017/%s.ivec' % lang
    name   = np.loadtxt(filename,usecols=[0],dtype='string')
    ivector = np.loadtxt(filename,usecols=range(1,401),dtype='float32')
    a = np.empty(len(ivector));a.fill(i)
    
    dev_names=np.append(dev_names,name)
    dev_ivectors = np.append(dev_ivectors, ivector,axis=0)
    dev_labels = np.append(dev_labels, a, axis=0)
    
#model and train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=400))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to categorical one-hot encoding

one_hot_labels = to_categorical(trn_labels, num_classes=5)

# Train the model, iterating on the data in batches of 32 samples
model.fit(trn_ivectors, one_hot_labels, epochs=70, batch_size=32)

#evaluate the model
#Convert labels to categorical one-hot encoding 

one_hot_labels_dev = to_categorical(dev_labels, num_classes=5)
scores = model.evaluate(dev_ivectors, one_hot_labels_dev)
#print scores

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
pred = model.predict(dev_ivectors,batch_size=32, verbose=10)
np.savetxt('pred.out', pred, delimiter=' ') 

pred_classes = model.predict_classes(dev_ivectors,batch_size=32, verbose=10)
np.savetxt('classes.out', pred_classes, delimiter=' ') 



#prob = model.predict_proba(dev_ivectors,batch_size=32, verbose=10)
#applying sigmoid function and normlise prob as something doesn't work with predict_proba function
import math
def sigmoid(x):return 1 / (1 + math.exp(-x))

prob = np.copy(pred)
i=0
for row in prob: 
  raw = [sigmoid(i) for i in row]
  norm = [float(i)/sum(raw) for i in raw]
  prob[int(i)]=norm
  i=i+1
np.savetxt('prob.out', prob, delimiter=' ') 
