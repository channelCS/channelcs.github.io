Edited on 14/01/18
#Build our first Neural Network for Audio Processing

Prerequisites:
* link1
* link2
 

A standard deep learning model passes the arrays of text or images dorectly to the Deep Neural Network or Convolution Neural etwork and the rest is done by the model itself.

As far as Audio is concerned, we first extract features which are then passed to the model for training.

In this tutorial, you will discover how to develop a multichannel convolutional neural network for Acoustic Scene Classification the NAR dataset.

##1. Unerstanding the Dataset
We shall be using the NAR dataset which can be downloaded from this [Link](https://team.inria.fr/perception/nard/) to the dataset. The data are freely accessible for scientific research purposes and for non-commercial applications.


NAR is a dataset  of audio recordings made with the humanoid robot Nao in real world conditions for sound recognition benchmarking.
####Audio Characteristics
There are certain parameters in audio which must be considered. These tell us about how and under what conditions were the recordings made for the dataset. The audio for the NAR Dataset has the following characteristics
* recorded with low-quality sensors (300 Hz – 18 kHz bandpass)
* suffering from typical fan noise from the robot’s internal hardware
* recorded in mutiple real domestic environments (no special acoustic charateristics, reverberations, presence of multiple sound sources and unknown locations)

####Dataset Characteristics
Now comes the details of the dataset files. These are important to consider as we have to convert everything in arrays and pass it on to the model.

The dataset is organized as follows:
* Each class is represented by a folder containing all the audio files labeled with the class.
* The name of a folder is the name of the class attached. The name of an audio file is “foldername$id.wav” where $id is an incremental identifier starting at 1.
* Each audio file is provided in a WAV format (mono signal, 48kHz sampling rate and 16 bits per sample).
* 42 differents class for 852 sounds have been recorded and organized.
* We shall consider four labels **Kitchen**, **Office**, **Nonverbal** and **Speech**.

##2. Dataset Manipulation

After downloading, we are going to extract it in a folder named ```NAR_dataset```. The ```tree``` looks something like
```
└───NAR_dataset
    ├───alarmfridge
    ├───alarmmicrowave
    ├───...
    ├───zipone
    └───ziptwo
```
We are going to make a file so that the directory structure changes to
| Scenarios | Classes                                                                                                                                                |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Kitchen   | Eating, Choking, Cuttlery, Fill a glass, Running the tap, Open/close a drawer,Move a chair, Open microwave,Close microwave, Microwave, Fridge, Toaster |
| Office    | Door Close, Open, Key, Knock, Ripped Paper, Zip, (another) Zip                                                                                         |
| Nonverbal | Fingerclap, Handclap, Tongue Clic                                                                                                                      |
| Speech    | 1,2,3,4,5,6,7,8,9,10, Hello, Left, Right, Turn, Move, Stop, Nao, Yes, No, What    

### 2.a. Making the config file
We are now making a config file for feature extraction in which we  place all the details about the dataset. We call it **_config.py_**.
There is no such rule of making a config fie as everything can be placed in a single place but that creates a lot of confusion when it comes to sharing codes between multiple developers.

```
import os
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
```
The ```CreateFolder``` comes in handy while creating multiple folders. It checks whether the folder is already present, if not it creates that folder.
```
from shutil import copytree
def MoveFolder(source,destination):
    try:
        copytree(source,destination)
    except:
        print("Oops! Folder already exists...")
        return
```
The ```MoveFolder``` function will be used when we want to move certain folders from one diretory to another.
```
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path.replace('\\', '/')
orig_dataset_path=dir_path+'/NAR_dataset/*'
```
We define where does all the audio files reside. In our case, we have put all our files under ```Nar_dataset``` directory.

```
kitchen_array=['alarmfridge', 'alarmmicrowave', 'chair', 'closemicrowave', 'cuttlery', 'drawer', 'eat', 'openmicrowave', 'strugling', 'tap', 'toaster', 'water']
nonverbal_array=['fingerclap', 'handclap', 'tongue']
office_array=['doorclose', 'doorkey', 'doorknock', 'dooropen', 'paper', 'zipone', 'ziptwo']
speech_array=['eight', 'five', 'four', 'hello', 'left', 'move', 'nao', 'nine', 'no', 'one', 'right', 'seven', 'six', 'stop', 'ten', 'three', 'turn', 'two', 'what', 'yes']
```
We now define, which folder shall be going under which ```Class Label```. This helps in moving the folders in the defined Label.

```
audio_folder='audios'
kitchen_folder   = audio_folder + '/Kitchen'
nonverbal_folder = audio_folder + '/Nonverbal'
office_folder    = audio_folder + '/Office'
speech_folder    = audio_folder + '/Speech'
```
We now define, the folder names for each Class Label. **Remember**, we have not made these folders yet.
the ```audios``` is the main folder and all the audio files shall reside in the same folder.

### 2.a. Making the file manipulator file
We are now going to make a file manipulator file the actual moving of files shall take place. We call it **_file\_manipulator.py_**.
#### 2.a.i. Moving folders under Class Labels
```
import glob
import config as cfg
import os
from string import digits
```
The ```glob``` function is used to read all files inside a specified folder.
We have also imported out ```config``` file here.
```
path=cfg.audio_folder

cfg.CreateFolder(cfg.audio_folder)
cfg.CreateFolder(cfg.kitchen_folder)
cfg.CreateFolder(cfg.nonverbal_folder)
cfg.CreateFolder(cfg.office_folder)
cfg.CreateFolder(cfg.speech_folder)
```
We define the path of our audio files which can directly be taken from the config file.
We are are making folders for our audio and all the class labels.

```
for f in glob.glob(cfg.orig_dataset_path):
    g=f.split('\\')[-1]
    if g in cfg.kitchen_array:
        cfg.MoveFolder(f, cfg.kitchen_folder+'/'+g)
    elif g in cfg.nonverbal_array:
        cfg.MoveFolder(f, cfg.nonverbal_folder+'/'+g)
    elif g in cfg.office_array:
        cfg.MoveFolder(f, cfg.office_folder+'/'+g)
    elif g in cfg.speech_array:
        cfg.MoveFolder(f, cfg.speech_folder+'/'+g)
```
This checks where should the folder of certain audio should go bsed in the config file and moves it.

The cuurent directory **structure** looks something like:
```
└───NAR_dataset
    ├───Kitchen
    ├───Nonverbal
    ├───Speech
    └───Office
```
#### 2.a.ii. Renaming Wav Files

Great WOrk! We now have moved all our subfolders under the specified class labels.
We are going to move and rename all our ```wav files``` so that they look something like:
```
ClassLabel_subtype_filename.wav
```
Example:
```
Kitchen_alarmfridge_alarmfridge1.wav
```


```
def move_files():
    x=os.listdir(path)
    print 'The folder has {} subfolders'.format(len(x))
    for folder in x:
        new_path=path+'/'+folder
        if os.path.isdir(new_path):
            y=os.listdir(new_path)
            if y == []:
                print 'Empty subfolder:',folder
            else:
                for file_ in y:
                    os.rename(new_path+'/'+file_,path+'/'+folder+'_'+file_)
                    if not os.listdir(new_path):
                        os.rmdir(new_path)
```
The function checks for non-empty subfolders and moves it. It then removes the original folder.
We have to run the function two times in order to get to the root directory level.
```
for f in glob.glob(path+'/*'):
    x=f.split('\\')[-1]
    if x[-4:]!='.wav':
        os.remove(path+'//'+x)
```     
This deletes the DS_Store files which are not required.
   
   
#### 2.a.iii. Generating the meta file
```
str1=''
arr1=[]
for f in glob.glob(path+'/*'):
    x=f.split('\\')[1]
    res = x.translate(None, digits).split('.')[0].split('_')[0]
    arr1.append(res)
    str1+='audio/'+x+'\t'+res+'\n'

file1 = open("meta.txt","w") 
file1.write(str1) 
file1.close()
```
This fetches all the files and puts them under the ```meta.txt``` such that each file corresponds to their class labels.
##3. Lets Code
###3.a. Feature Extraction
In case of Audio we flatten the data and pass it to the  layers. We need to encapsulate the statistics of sound and make our model learn faster. IN our case we are using :
 * MEL filterbanks : Create a Filterbank matrix to combine FFT bins into Mel-frequency bins. 
 * CQT(constant q transform) : THe Constant-Q-Transform (CQT) is a time-frequency representation where the frequency bins are geometrically spaced and the so called Q-factors (ratios of the center frequencies to bandwidths) of all bins are equal.The CQT essentially a wavelet transform, which means that the frequency resolution is better for low frequencies and the time resolution is better for high frequencies.[4]
 * LOG-MEL(Logarithm - mel) : We take logarithm of the Filterbank matrix.
 * LOG-MFCC(Logarithm - MFCC ) : A widely used metric for describing timbral characteristics based on the Mel scale. Implemented according to Huang [1], Davis [2], Grierson [3] and the librosa library.
###3.b. Simple DNN Mode
We are going to make a simple dnn model and pass on certain parameters which are required for the model.
```python
#Our files
import config as cfg
import features as F
import apnahat as H

#Python modules
import time
import csv
import cPickle

#Data managing modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, accuracy_score

#Deep Learning Modules
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical

```
Whenever we work with machine learning algorithms that use a stochastic process (e.g. random numbers), it is a good idea to set the random number seed.

This is so that you can run the same code again and again and get the same result. 
You can initialize the random number generator with any seed you like      
```python
np.random.seed(1234)
```

Each feature has a separate dimension, we create a function which returns the function based on the feature.
```python
def get_dimension(feature):
        return {
            "cqt"               :80,
            "logmel_kong"       :40,
            "logmel_lib_delta"  :60,
            "mel"               :40,
         }.get(feature, 1000) 
```
As we have used different different features, we are going to call a single function to select the function call
```python
def get_feature(feature):
        return {
            "cqt"                :F.feature1(),
            "logmel_kong"        :F.feature2(),
            "logmel_lib_delta"   :F.feature3(),
            "mel"                :F.feature4(),
           }.get(feature, 1000) 
```
We make a function call for getting the dimension and feature.
```python
dimension1 = get_dimension(feature)
dimension2 = dimension1*10
fe_fd,feature_text=get_feature(feature)
print "Feature",feature_text
```
We define all our ```hyperparameters```. Configuring neural networks is difficult because there is no good theory on how to do it.

We must be systematic and explore different configurations and  understand what is going on for a given predictive modeling problem.
```python
input_neurons=200
dropout=0.1
act1='linear'
act2='relu'
act3='sigmoid'
act4='softmax'
epochs=20
batchsize=20
agg_num=10
hop=10
feature="cqt"
num_classes=4
```
We now make a separate function for using the ```meta``` file as a base for calling all the features of the **audio** files. This functions returns a 3d array as it comes in handy when handling Convolution Neural Networks. 
```python
def GetAllData(fe_fd, csv_file, agg_num, hop):
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # init list
    X3d_all = []
    y_all = []
    i=0
    for li in lis:
        # load data
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        #i+=1
        #print i
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        if i%100==0:
            print "Files Loaded",i
        X3d = H.mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ cfg.lb_to_id[lb] ] * len( X3d )
    
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all
```
We call the function to return a 3d array of train X and  1d array of train Y. We now reshape our 3d array into 1d.
```python
tr_X, tr_y = GetAllData( fe_fd, cfg.meta_csv, agg_num, hop )
tr_X=tr_X.reshape(tr_X.shape[0],tr_X.shape[1]*tr_X.shape[2])
```
**Altering the feature arrays**

We are using a single function from the [hat](http://www.example.com) module. We are going to make a separte model for that and call it **_apnahat.py_**. The function takes a 2d array as input and returns a 3d array. We shall be using this to pass into our model.
```python
import numpy as np
def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)
```


####3.b.ii. The DNN Model
We are now going to make a funtion for our model which returns a compiled model.
```python
def prepare_model():
    model = Sequential()
    model.add(Dense(input_neurons, input_dim = dimension2, activation=act1))
    lr=LeakyReLU(alpha=.001)
    model.add(lr)
    model.add(Dropout(dropout))
    model.add(Dense(input_neurons, activation=act2))
    model.add(Dropout(dropout))
    model.add(Dense(input_neurons, activation=act3))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation=act4))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
```
We are goig for cross validation. In our case we are going to do a 2-fold cross validation.
```python
kf = KFold(len(tr_X),2,shuffle=True,random_state=42)
results=[]    
for train_indices, test_indices in kf:
    train_x = [tr_X[ii] for ii in train_indices]
    train_y = [tr_y[ii] for ii in train_indices]
    test_x  = [tr_X[ii] for ii in test_indices]
    test_y  = [tr_y[ii] for ii in test_indices]
    train_y = to_categorical(train_y,num_classes=4)
    test_y = to_categorical(test_y,num_classes=4) 
    
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    test_x=np.array(test_x)
    test_y=np.array(test_y)
    print "All arrays loaded"
    
    #get compiled model
    lrmodel=prepare_model()
    #see the model
    print lrmodel.summary()
    #fit the model
    lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
    
    #make prediction
    pred=lrmodel.predict(test_x, batch_size=32, verbose=0)

    pred = [ii.argmax()for ii in pred]
    test_y = [ii.argmax()for ii in test_y]

    results.append(accuracy_score(pred,test_y))
    print accuracy_score(pred,test_y)
    jj=str(set(list(test_y)))
    print "Unique in test_y",jj
print "Results: " + str( np.array(results).mean() )
print classification_report(np.array(test_y),pred).split('\n')
```



<!--
####2. The config file
The following lines of codes must be added in the config file:

```python
# We define where does all the audio files reside
wav_fd = 'trim'

# These are the folders where various features will be extracted
fe_cqt_fd 			 = 'Fe/cqt'
fe_logmel_kong_fd 	 = 'Fe/logmel_kong'
fe_logmel_libd_fd 	 = 'Fe/logmel_lib_delta'
fe_mel_fd 			 = 'Fe/mel

```

* The paths of audio files
-->
<!--
##2. Extracting Features

##3. Making models

##4. Understanding Features

##5. Implementing Feature Inside Model

##6. Running Code

##7. Understanding Accuracy
-->

####4. References

[1] X. Huang, A. Acero, and H.-W. Hon, Spoken Language Processing: A Guide to Theory, Algorithm, and System Development. Upper Saddle River, NJ, USA: Prentice Hall PTR, 1st ed., 2001.

[2] S. Davis and P. Mermelstein, “Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,” Acoustics, Speech and Signal Processing, IEEE Transactions on, vol. 28, pp. 357–366, Aug 1980.

[3] M. Grierson, “Maximilian: A cross platform c++ audio synthesis library for artists learning to program.,” in Proceedings of International Computer Music Conference, 2010.

[4] lidy2016cqt,"CQT-based convolutional neural networks for audio scene classification and domestic audio tagging," in IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events (DCASE 2016), Budapest, Hungary, Tech. Rep, 2016.
}
