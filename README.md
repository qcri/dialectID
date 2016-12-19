#Automatic Dialect Detection Task 

This code reflects the work done in Automatic Dialect Detection in Arabic Broadcast Speech InterSpeech 2016 paper

This also has a simple baseline system for Automatic Dialect Detection task in vardial2017 

# Requirements
* Python (tested with v.2.7.5)
* Multi-Class (http://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)

# Provided data:
* We provide data for five Arabic dialects; named Egyptain (EGY), Levantine (LAV), Gulf (GLF), North African (NOR), and Modern Standard Arabic (MSA)

* The data have been generated from brodcast news.

* For each dialect, there are four files:
** $dialect.words; this is the lexical features generated using LVCSR
** $dialect.phones; this is the pheoneme sequence using phoneme recognition system
** $dialect.phone_duration; this is the phoneme sequence, and the duration in milli seconds for each phone, for example w_030 means phone w for 30 milli seconds
** $dialect.ivec; this is the bottle neck ivector feature vector. It has a fixed length of 400 per uttrance.

# Sample code 

Run 'run.sh' this can be used to show an example of the code and the data
* features=phones, you can use words, duration  or ivectors
* context=6, in some features, less context is enough 
* regularization parameters can be optimised for better gain 
* sysmtem combination can be explored as well 


# Citing

This data and the baseline system  is described in [this](http://www.cstr.ed.ac.uk/downloads/publications/2016/is2016-automatic-dialect-detection.pdf) paper:

    @inproceedings{ali2016automatic,
      author={Ali, Ahmed and Dehak, Najim and Cardinal, Patrick and Khurana, Sameer and Yella, Sree Harsha and Glass, James and Bell, Peter and Renals, Steve},
      title={Automatic Dialect Detection in Arabic Broadcast Speech},
      booktitle={Interspeech, San Francisco, USA},
      pages={2934--2938},
      year={2016}
    }

