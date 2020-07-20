""""
Code credits to code for IJCNN 2018 submission: https://github.com/ynulonger/ijcnn

"""

import scipy.io as sio
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle

np.random.seed(0)

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal_without_transition(data,label,label_index,window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if((len(data[start:end]) == window_size)):
            if(start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index])) # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels

def apply_mixup(dataset_file,window_size,label,yes_or_not): # initial empty label arrays
    print("Processing",dataset_file,"..........")
    data_file_in = sio.loadmat(dataset_file)
    data_in = data_file_in["data"].transpose(0,2,1)
    #0 valence, 1 arousal, 2 dominance, 3 liking
    if label=="arousal":
        label=1
    elif label=="valence":
        label=0
    label_in= data_file_in["labels"][:,label]>5
    label_inter	= np.empty([0]) # initial empty data arrays
    data_inter_cnn	= np.empty([0,window_size, 9, 9])
    data_inter_rnn	= np.empty([0, window_size, 32])
    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0,trials):
        if yes_or_not=="yes":
            base_signal = (data_in[trial,0:128,0:32]+data_in[trial,128:256,0:32]+data_in[trial,256:384,0:32])/3
        else:
            base_signal = 0
        data = data_in[trial,384:8064,0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0,60):
            data[i*128:(i+1)*128,0:32]=data[i*128:(i+1)*128,0:32]-base_signal
        label_index = trial
        #read data and label
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in,label_index,window_size)
        # rnn data process
        data_rnn    = data. reshape(int(data.shape[0]/window_size), window_size, 32)
        # append new data and label
        data_inter_rnn  = np.vstack([data_inter_rnn, data_rnn])
        label_inter = np.append(label_inter, label)
    '''
    print("total rnn size:", data_inter_rnn.shape)
    print("total label size:", label_inter.shape)
    '''
    # shuffle data
    index = np.array(range(0, len(label_inter)))
    np.random.shuffle( index)
    shuffled_data_rnn	= data_inter_rnn[index]
    shuffled_label 	= label_inter[index]
    return shuffled_data_rnn,shuffled_label,record

if __name__ == '__main__' :
    begin = time.time()
    print("time begin:",time.localtime())
    dataset_dir		=   "/home/bsipl_5/experiment/data_preprocessed_matlab/"
    window_size		=	128
    output_dir		=   "./deap_shuffled_data/"
    label_class = "valence"  
    suffix = "yes"     
    
    # get directory name for one subject
    record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir,task))]
    output_dir = output_dir+suffix+"_"+label_class+"/"
    if os.path.isdir(output_dir)==False:
        os.makedirs(output_dir)
    # print(record_list)

    for record in record_list:
        file = os.path.join(dataset_dir,record)
        shuffled_rnn_data,shuffled_label,record = apply_mixup(file, window_size,label_class,suffix)
        output_data_rnn = output_dir+record+"_win_"+str(window_size)+"_rnn_dataset.pkl"
        output_label= output_dir+record+"_win_"+str(window_size)+"_labels.pkl"

        with open( output_data_rnn, "wb") as fp:
            pickle.dump(shuffled_rnn_data, fp, protocol=4)
        with open(output_label, "wb") as fp:
            pickle.dump(shuffled_label, fp)
        end = time.time()
        print("end time:",time.localtime())
        print("time consuming:",(end-begin))
        
        # break
