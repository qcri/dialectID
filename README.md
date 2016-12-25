# Automatic Arabic Dialect Detection Task 

This code reflects the work described in the InterSpeech'2016 paper on Automatic Dialect Detection in Arabic Broadcast Speech.

It also contains a baseline system for the VarDial'2017 shared task on Arabic Dialect Identification.

# Requirements
* Python (tested with v.2.7.5)
* Multi-class SVM (http://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)

# Provided data:
* We provide data for five Arabic dialects: Egyptian (EGY), Levantine (LAV), Gulf (GLF), North African (NOR), and Modern Standard Arabic (MSA).

* The data comes from broadcast news.

***VarDial'2017 shared task*** shared data, and features.
* The baseline for VarDial'2017 is using data/train.vardial2017/ and data/dev.vardial2017/ for training and development ***default***
* For each dialect, there are two features files:
* $dialect.words -- lexical features generated using LVCSR- generated using QCRI MGB-2 submission.
* $dialect.ivec -- i-vector based on bottleneck features, with a fixed length of 400 per utterance.
* wav.lst -- link to the original audio files; WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz.
* ***Baseline***-- bottleneck iVectors 57.28% accuracy and lexical features 48.43%. 

***InterSpeech'2016 paper*** shared data.
* To reproduce the results in InterSpeech'2016, the script should point to data/train.IS2016/ and data/test.IS2016/ for training and testing.
* $dialect.words -- lexical features generated using LVCSR;
* $dialect.ivec -- i-vector based on bottleneck features, with a fixed length of 400 per utterance.
* $dialect.phones -- phoneme sequence from an automatic phoneme recognition system.
* $dialect.phone_duration -- phoneme sequence, and the duration in milliseconds for each phone, e.g., w_030 means phone w for 30 milliseconds.



# Sample code 

Run 'run.sh' for an example of the code and the data
* features=phones -- you can use words, phones or ivectors;
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
