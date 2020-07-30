# Code for Multi-level Features Guided Capsule Network （MLF-CapsNet）
This repository contains the Keras implementation for the paper: "Multi-channel EEG-based Emotion Recognition via a Multi-level Features Guided Capsule Network"
## About the paper
* Title: [Multi-channel EEG-based Emotion Recognition via a Multi-level Features Guided Capsule Network](https://www.sciencedirect.com/science/article/pii/S0010482520302663?via%3Dihub)
* Authors: Yu Liu, [Yufeng Ding](https://github.com/2018110060ding), Chang Li, Juan Cheng, Rencheng Song, Feng Wan, Xun Chen
* Institution: Hefei University of Technology
* Published in: 2020 Computers in Biology and Medicine (CBM)
## Instructions
* Before running the code, please download the DEAP dataset, unzip it and place it into the right directory. The dataset can be found [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html). Each .mat data file contains the EEG signals and consponding labels of a subject. There are 2 arrays in the file: **data** and **labels**. The shape of **data** is (40, 40, 8064). The shape of **label** is (40,4). 
* Please run the deap_pre_process.py to Load the origin .mat data file and transform it into .pkl file.
* Using capsulenet-multi-gpu.py to train and test the model (10-fold cross-validation), result of 10 folds will be saved in a .xls file.
* count_accuracy_deap.py is used to calculate the final accuracy of the model.
* The usage on DREAMER dataset is the same as above. The DREAMER dataset can be found [here](https://zenodo.org/record/546113/accessrequest). 
## Requirements
+ Pyhton3.5
+ tensorflow (1.3.0 version)
+ keras (2.2.4 version)

If you have any questions, please contact yfding@mail.hfut.edu.cn

## Reference
* [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
* [ynulonger/ijcnn](https://github.com/ynulonger/ijcnn)
