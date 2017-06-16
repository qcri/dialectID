#!/usr/bin/python -tt

import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='generate normalized probablity out of the score')
parser.add_argument('scoreFile', help='input SVM score')
parser.add_argument('propFile', help='output norm prob')

args = parser.parse_args()

# We use sigmoid to convert score to probablities (un-normalised)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

file_w = open(args.propFile,"w") 
with open(args.scoreFile) as f:
    for line in f:
       id=line.split()[0]
       _class=line.split()[1]
       score=map(float,line.split()[2:])
       prob = [sigmoid(i) for i in score]
       norm = [float(i)/sum(prob) for i in prob]
       norm_formatted = [ '%.6f' % elem for elem in norm ]
       results=id + " " + _class + " " + ' '.join(norm_formatted) + '\n'
       file_w.write(results) 
file_w.close()         
        

