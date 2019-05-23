import pickle
import numpy as np
import pandas as pd
import os


def deap_preprocess(data_file,dimention):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"

    # data_file    =sys.argv[1]
    # arousal_or_valence = sys.argv[2]
    # with_or_without = sys.argv[3]

    #data_file = 's22'
    arousal_or_valence = dimention

    with_or_without = 'yes'

    dataset_dir = "/home/bsipl_5/experiment/ijcnn-master/deap_shuffled_data/" + with_or_without + "_" + arousal_or_valence + "/"

    ###load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)
        # print("loaded shape:",labels.shape)
    lables_backup = labels
    one_hot_labels = np.array(list(pd.get_dummies(labels)))

    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]
    datasets = rnn_datasets

    datasets = datasets.reshape(-1, 128, 32, 1).astype('float32')
    labels = labels.astype('float32')

    """
    # generate training sets and testing sets
    x_train = rnn_datasets[0:2000]
    y_train = labels[0:2000]
    x_test = rnn_datasets[2000:2400]
    y_test = labels[2000:2400]
    """

    #return (x_train, y_train), (x_test, y_test)

    return datasets , labels