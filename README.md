#Automatic Dialect Detection Task 

This code reflects the work described in the InterSpeech'2016 paper on Automatic Dialect Detection in Arabic Broadcast Speech.

It also contains a simple baseline system for the VarDial'2017 shared task on Arabic Dialect Identification.

# Requirements
* Python (tested with v.2.7.5)
* Multi-class SVM (http://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)

# Provided data:
* We provide data for five Arabic dialects: Egyptian (EGY), Levantine (LAV), Gulf (GLF), North African (NOR), and Modern Standard Arabic (MSA).

* The data comes from broadcast news.

* For each dialect, there are four files:
* $dialect.words -- lexical features generated using LVCSR;
* $dialect.phones -- phoneme sequence from an automatic phoneme recognition system;
* $dialect.phone_duration -- phoneme sequence, and the duration in milliseconds for each phone, e.g., w_030 means phone w for 30 milliseconds;
* $dialect.ivec -- i-vector based on bottleneck features, with a fixed length of 400 per utterance.

# Sample code 

Run 'run.sh' for an example of the code and the data
* features=phones -- you can use words, duration or ivectors;
* context=6 -- for some features, less context might be enough;
* NOTE 1: The regularization parameters can be optimized for better performance.
* NOTE 2: System combination can be explored as well.


# Citing

This data and the baseline system are described in [this](http://www.cstr.ed.ac.uk/downloads/publications/2016/is2016-automatic-dialect-detection.pdf) paper:

    @inproceedings{ali2016automatic,
      author={Ali, Ahmed and Dehak, Najim and Cardinal, Patrick and Khurana, Sameer and Yella, Sree Harsha and Glass, James and Bell, Peter and Renals, Steve},
      title={Automatic Dialect Detection in Arabic Broadcast Speech},
      booktitle={Interspeech},
      address={San Francisco, CA, USA}
      pages={2934--2938},
      year={2016}
    }
