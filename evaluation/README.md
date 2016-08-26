====
## 
# README.md
#
This folder has the evaluation code for dialect detection task
precision and recall will be used for evaluation 
Both hypothesis and reference should have only one utterance per line and the langue ID should be the last field. Look at sample files hyp and ref
Supported dialects/languages are: Egyptian(EGY), Gulf(GLF), Levantine(LEV), Modern Standard Arabic(MSA) or North-African(NOR):
Enclosed here sample: ./eval.py ref hyp

***This code doesn't assume any indexing, so both files must have the same number of lines with the same order***
