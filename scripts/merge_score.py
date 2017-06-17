#!/usr/bin/python -tt

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='combine two systems or more')
parser.add_argument('prob_sys1', help='prob from system1')
parser.add_argument('prob_sys2', help='prob from system2')
parser.add_argument("-sys3", help='probabilities from system2', type=str, default=False)


args = parser.parse_args()

data_sys1  = np.loadtxt(args.prob_sys1, delimiter=' ', usecols=range(2,7))
data_sys2  = np.loadtxt(args.prob_sys2, delimiter=' ', usecols=range(2,7))

if args.sys3: 
  data_sys3  = np.loadtxt(args.sys3, delimiter=' ', usecols=range(2,7))
  
with open(args.prob_sys1) as f:
    id_list = [line.split()[0] for line in f]

overall=data_sys1+data_sys2+data_sys3

i=0
for line in overall:
    index=np.argmax(line)
    print id_list[i],index+1,  str(line)[1:-1]
    i=i+1


