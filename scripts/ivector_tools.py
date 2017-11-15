import numpy as np
import os, re,gzip,struct
import scipy as sc

def read_ivector_text(file):
    
    file = open(file,'r')
    
    ivector = []
    for line in file:
        a = line.split('[ ')
        b=a[1].split(' ]')
        c= b[0].split(' ')
        dim = np.size(c)
        d=np.array(c)
        e=d.astype('float32')
        ivector.append(e)
    
    print np.shape(ivector)
    file.close()

    return ivector

def read_ivector_binary(file, point):

    ivecfile = open(file,'r')
    
    ivecname = ivecfile.read(point)
    #print ivecname

    #binary flag
    binary=ivecfile.read(2)
    #print binary
    #type flag (FV for 4 byteor DV for 8)
    type=ivecfile.read(3)
    if type == 'FV ': 
        sample_type='float32'
    if type =='DV ': 
        sample_type = 'float64'

    temp=ivecfile.read(1) #int-size
    vec_size=struct.unpack('<i',ivecfile.read(4))[0]
    ivector = np.fromfile(ivecfile,sample_type,vec_size)
    ivecfile.close()

    return ivector

def read_ivector_key(file):
    spkid = []
    ivec_pointer=[]
    total_num = 0
    scpfile = open(file,'r')
    for line in scpfile:
        #print line
        temp = re.split(' |:|\n',line)
        spkid.append(temp[0])
        ivec_pointer.append(int(temp[2]))
        total_num +=1
    
    return spkid, ivec_pointer,total_num

def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line,2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)

    return norm_mat

def lda(mat,label):
    # mat = observation x dim ( for example, 8x600 for 8 obs and 600dim ivector)
    # label = num_utts (for example, [2,4,2] for 8 observations)

    #Sw = np.inner(mat.transpose(), mat.transpose())
    Sw = np.dot(mat.transpose(), mat)
    mu_c=[]
    pre_iter2 = 0
    for iter1, iter2 in enumerate(label):
        idx = np.arange(pre_iter2,pre_iter2+iter2)
        pre_iter2 += iter2
        temp = mat[idx]
        mu_c.append(np.math.sqrt(iter2) * np.mean(temp,axis=0))

    mu_c = np.array(mu_c)

    Sb = np.dot(mu_c.transpose(),mu_c)
    [D, V] = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    descend_idx =  (-D).argsort()
    V= V[:,descend_idx]
    V = length_norm(V)
    
    return V

def lda2(mat,label):
    # mat = observation x dim ( for example, 8x600 for 8 obs and 600dim ivector)
    # label = index num for all observations (for example, [0,1,1,2,0,2,1,0] for 8 observations with 3 class)

    #Sw = np.inner(mat.transpose(), mat.transpose())
    Sw = np.dot(mat.transpose(), mat)
    
    mu_c=[]
    pre_iter2 = 0
    for iter1, iter2 in enumerate(np.unique(label)):
        temp = mat[label==iter1,:]
        mu_c.append(np.math.sqrt(temp.shape[0]) * np.mean(temp,axis=0))
    mu_c = np.array(mu_c)

    Sb = np.dot(mu_c.transpose(),mu_c)
    [D, V] = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    descend_idx =  (-D).argsort()
    V= V[:,descend_idx]
    
    return V

def load_ivector_fromtextark(foldername,num_arks):
#loading ivectors from text-ark files
    
    spkid, point, total_num=read_ivector_key(foldername+'ivector.scp')
    ivec1 = []
    spk_ivectors = []
    ivec1 = read_ivector_text(foldername+'ivector.1.ark')
    for iter1 in np.arange(2,num_arks+1):
        temp = []
        temp = read_ivector_text(foldername+'ivector.'+str(iter1)+'.ark')
        ivec1=np.append(ivec1,temp,axis=0)
    spk_ivectors = ivec1
    print 'total',total_num,'ivector were saved on spk_ivector variable(shape is',np.shape(spk_ivectors),' )'
    return spk_ivectors, spkid, point, total_num
