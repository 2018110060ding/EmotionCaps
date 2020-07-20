import scipy.io as sio
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle
import re

np.random.seed(0)

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          0,          0,          0, 	        0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[0],    0,          data[13],   0,          0,          0       )
    data_2D[2] = (data[1],  0,          0,          0,          0,          0,          data[11],   0,          data[12])
    data_2D[3] = (0,        data[3],    0,          0,          0,          0,          0,          data[10],   0       )
    data_2D[4] = (data[4],  0,          data[2],    0,          0,          0,          0,          0,          data[9] )
    data_2D[5] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
    data_2D[6] = (data[5],  0,          0,          0,          0,          0,          0,          0,          data[8] )
    data_2D[7] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[6],    0,          data[7],    0,          0,          0       )
    # return shape:9*9
    return data_2D

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 14])
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

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],9,9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize( data_1Dto2D(dataset_1D[i]))
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal(data, label, label_index, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index]))  # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels

def segment_baseline(base, window_size):
    # get data file name
    for (start, end) in windows(base, window_size):
        # print(data.shape)
        if ((len(base[start:end]) == window_size)):
            if (start == 0):
                segments_ = base[start:end]
                segments_ = np.vstack([segments_, base[start:end]])
            else:
                segments_ = np.vstack([segments_, base[start:end]])
    return segments_

if __name__ == '__main__':
    begin = time.time()
    print("time begin:", time.localtime())
    label_class = "arousal"  # sys.argv[1]     # arousal/valence/dominance
    subs = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']                # sys.argv[2]
    for sub in subs:
        debase = 'no'
        dataset_dir = '/home/bsipl_5/experiment/Data/dreamer/'
        dataset_dir1 =dataset_dir+'stimuli(m,14)/'+sub+'_stimuli'
        data_file1 = sio.loadmat(dataset_dir1 + ".mat")
        dataset_dir2 =dataset_dir+'label_'+label_class+'(18,1)/'+sub+'_'+label_class+'_label'  #arousal/valence/dominance
        data_file2 = sio.loadmat(dataset_dir2 + ".mat")
        dataset_dir3 =dataset_dir+'baseline(m,14)/'+sub+'_baseline'       #
        data_file3 = sio.loadmat(dataset_dir3 + ".mat")
        output_dir = '/home/bsipl_5/experiment/ijcnn-master/dreamer_shuffled_data/' + debase + '_'+label_class + '/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        window_size = 128

        # load label
        label_in =data_file2['label']

        label_inter = np.empty([0])
        data_inter_cnn	= np.empty([0,window_size, 9, 9])
        data_inter_rnn	= np.empty([0, window_size, 14])
        baseline_inter = np.empty([0, window_size, 14])
        base_signal_ = np.zeros([window_size, 14])
        base_signal = np.empty([18, window_size, 14])
        # load dataset
        dataset_in = data_file1

        key_list = [key for key in data_file1.keys() if key.startswith('__') == False]
        print(key_list)
        key_list_rearrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key in key_list:
            index = int(re.findall('\d+', key)[0]) - 1
            key_list_rearrange[index] = key
        print(key_list_rearrange)

        key_list2 = [key for key in data_file3.keys() if key.startswith('__') == False]
        print(key_list2)
        key_list_rearrange2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key in key_list2:
            index = int(re.findall('\d+', key)[0]) - 1
            key_list_rearrange2[index] = key
        print(key_list_rearrange2)

        for key in key_list_rearrange2:
            if key.startswith('__') == False:  # if array is EEG than do as follow
                print("Processing ",  key, "..........")
                baseline = data_file3[key]
                label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key
                # data = norm_dataset(data)  # normalization
                print('shape of this EEG: ', baseline.shape)
                baseline = segment_baseline(baseline, window_size)
                base_segment_set = baseline.reshape(int(baseline.shape[0] / window_size), window_size, 14)
                print('segment number of this EEG: ', base_segment_set.shape[0])
                baseline_inter = np.vstack([baseline_inter, base_segment_set])
                print(baseline_inter.shape)

        for k in range(1, 19):
            for i in range(1, 62):
                base_signal_ += baseline_inter[k * i - 1]
            base_signal_ = base_signal_ / 61
            base_signal[k-1] = base_signal_
        print(base_signal.shape)

        count = 1
        # traversing 18 EEGs of one experiment/session
        for key in key_list_rearrange:
             if key.startswith('__') == False:  # if array is EEG than do as follow
              print("Processing ", count, key, "..........")
              count = count + 1
              data = dataset_in[key]
              label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key

              if debase =='yes':
                  for m in range(0, data.shape[0]//128):
                      data[m * 128:(m + 1) * 128, 0:14] = data[m * 128:(m + 1) * 128, 0:14] - base_signal[label_index]
              data = norm_dataset(data)  # normalization
              print('shape of this EEG: ', data.shape)
              data, label = segment_signal(data, label_in, label_index, window_size)
              # cnn data process
              data_cnn = dataset_1Dto2D(data)
              data_cnn = data_cnn.reshape(int(data_cnn.shape[0] / window_size), window_size, 9, 9)
              # rnn data process
              data_rnn = data.reshape(int(data.shape[0] / window_size), window_size, 14)
              # append new data and label
              data_inter_cnn = np.vstack([data_inter_cnn, data_cnn])
              data_inter_rnn = np.vstack([data_inter_rnn, data_rnn])
              label_inter = np.append(label_inter, label)
              print("total cnn size:", data_inter_cnn.shape)
              print("total rnn size:", data_inter_rnn.shape)
              print("total label size:", label_inter.shape)

        # shuffle data
        index = np.array(range(0, len(label_inter)))
        np.random.shuffle(index)
        shuffled_data_cnn = data_inter_cnn[index]
        shuffled_data_rnn = data_inter_rnn[index]
        shuffled_label = label_inter[index]

        output_data_cnn = output_dir + sub + "_cnn_dataset.pkl"
        output_data_rnn = output_dir + sub + "_rnn_dataset.pkl"
        output_label = output_dir + sub + '_'+label_class+'_labels.pkl'   #arousal/valence/dominance
        with open(output_data_cnn, "wb") as fp:
            pickle.dump(shuffled_data_cnn, fp, protocol=4)
        with open(output_data_rnn, "wb") as fp:
            pickle.dump(shuffled_data_rnn, fp, protocol=4)
        with open(output_label, "wb") as fp:
            pickle.dump(shuffled_label, fp)
        #break
        end = time.time()
        print("end time:", time.asctime(time.localtime(time.time())))
        print("time consuming:", (end - begin))