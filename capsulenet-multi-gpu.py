"""
Keras implementation of Multi-level Features Guided Capsule Network (MLF-CapsNet).
This file trains a MLF-CapsNet on DEAP/DREAMER dataset with the parameters as mentioned in paper.
We have developed this code using the following GitHub repositories:
- Xifeng Guo's CapsNet code (https://github.com/XifengGuo/CapsNet-Keras)

Usage:
       python capsulenet-multi-gpu.py --gpus 2

"""

from keras import backend as K
from keras import layers, models, optimizers,regularizers
from capsulelayers import CapsuleLayer, PrimaryCap, Length

K.set_image_data_format('channels_last')

import pandas as pd
import time
import pickle
import numpy as np


def deap_load(data_file,dimention,debaseline):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"
    arousal_or_valence = dimention
    with_or_without = debaseline # 'yes','not'
    dataset_dir = "/home/bsipl_5/experiment/ijcnn-master/deap_shuffled_data/" + with_or_without + "_" + arousal_or_valence + "/"

    ###load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    datasets = rnn_datasets.reshape(-1, 128, 32, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets , labels

def dreamer_load(sub,dimention,debaseline):
    if debaseline == 'yes':
        dataset_suffix = "f_dataset.pkl"
        label_suffix = "_labels.pkl"
        dataset_dir = "/home/bsipl_5/experiment/Data/data_pre(-base)/" + dimention + "/"
    else:
        dataset_suffix = "_rnn_dataset.pkl"
        label_suffix = "_labels.pkl"
        dataset_dir = '/home/bsipl_5/experiment/ijcnn-master/dreamer_shuffled_data/' + 'no_' + dimention + '/'

    ###load training set
    with open(dataset_dir + sub + dataset_suffix, "rb") as fp:
        datasets = pickle.load(fp)
    with open(dataset_dir + sub + '_' + dimention + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    labels = labels > 3
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    datasets = datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    datasets = datasets.reshape(-1, 128, 14, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets , labels

def CapsNet(input_shape, n_class, routings, model_version,lam_regularize):
    """
    A Capsule Network .
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer, 对DEAP，kernel_size=9；对DREAMER，kernel_size=6
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1',kernel_regularizer=regularizers.l2(lam_regularize))(x) #kernel_size=9

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    # 对DEAP，kernel_size=9；对DREAMER，kernel_size=6
    # 对CapsNet，strides=2，pading=‘valid’；对MLF-CapsNet，stides=1，padding='same'
    if model_version == 'v0':
        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid',lam_regularize = lam_regularize,model_version =model_version )
    else:
        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=1, padding='same',lam_regularize = lam_regularize,model_version =model_version )          #kernel_size=9

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps', lam_regularize = lam_regularize)(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    y = layers.Input(shape=(n_class,))
    train_model = models.Model([x, y], out_caps)
    eval_model = models.Model(x, out_caps)


    return train_model, eval_model

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model, data, args,fold):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/' + 'log_fold'+str(fold)+'.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs_fold'+str(fold),
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}_fold'+str(fold)+'.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (1.0 ** epoch))

    #EarlyStop = callbacks.EarlyStopping(monitor='val_capsnet_acc', patience=5)
    # compile the model
    model.compile(optimizer= optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay, EarlyStop])
    """

    '''
    # Training with validation set
    model.fit([x_train, y_train], y_train ,  batch_size=args.batch_size, epochs=args.epochs,verbose = 1,
              validation_split= 0.1 , callbacks=[log, tb, checkpoint, lr_decay])
    '''

    # Training without validation set
    model.fit([x_train, y_train],y_train, batch_size=args.batch_size, epochs=args.epochs,
                callbacks=[log, tb, lr_decay])


    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model

time_start_whole = time.time()

dataset_name = 'deap' #'deap' # dreamer
subjects = ['s21','s22','s23','s24','s25','s26','s28','s29','s30','s31','s32']  #  ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16']#,'s05']#,'s06','s07','s08']#,'s09','s10','s11','s12','s13','s14','s15','s16'，'s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28',]
#subjects = ['s01'] #'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
dimentions = ['dominance']#,'arousal','dominance']
debaseline = 'yes' # yes or not
tune_overfit = 'tune_overfit'
model_version = 'v2' # v0:'CapsNet', v1:'MLF-CapsNet(w/o)', v2:'MLF-CapsNet'


if __name__ == "__main__":
    for dimention in dimentions:
        for subject in subjects:
            import numpy as np
            import tensorflow as tf
            import os
            from keras import callbacks
            from keras.utils.vis_utils import plot_model
            from keras.utils import multi_gpu_model

            # setting the hyper parameters
            import argparse
            parser = argparse.ArgumentParser(description="Capsule Network on " + dataset_name)
            parser.add_argument('--epochs', default=40, type=int)  # v0:20, v2:40
            parser.add_argument('--batch_size', default=100, type=int)
            parser.add_argument('--lam_regularize', default=0.0, type=float,
                                help="The coefficient for the regularizers")
            parser.add_argument('-r', '--routings', default=3, type=int,
                                help="Number of iterations used in routing algorithm. should > 0")
            parser.add_argument('--debug', default=0, type=int,
                                help="Save weights by TensorBoard")
            parser.add_argument('--save_dir', default='./result_'+ dataset_name + '/sub_dependent_'+ model_version +'/') # other
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            parser.add_argument('--lr', default=0.00001, type=float,
                                help="Initial learning rate")  # v0:0.0001, v2:0.00001
            parser.add_argument('--gpus', default=2, type=int)
            args = parser.parse_args()

            print(time.asctime(time.localtime(time.time())))
            print(args)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            if dataset_name == 'dreamer':          # load dreamer data
                datasets,labels = dreamer_load(subject,dimention,debaseline)
            else:  # load deap data
                datasets,labels = deap_load(subject,dimention,debaseline)

            args.save_dir = args.save_dir + '/' + debaseline + '/' + subject + '_' + dimention + str(args.epochs)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            fold = 10
            test_accuracy_allfold = np.zeros(shape=[0], dtype=float)
            train_used_time_allfold = np.zeros(shape=[0], dtype=float)
            test_used_time_allfold = np.zeros(shape=[0], dtype=float)
            for curr_fold in range(fold):
                fold_size = datasets.shape[0] // fold
                indexes_list = [i for i in range(len(datasets))]
                #indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
                split = np.array(split_list)
                x_test = datasets[split]
                y_test = labels[split]

                split = np.array(list(set(indexes_list) ^ set(split_list)))
                x_train = datasets[split]
                y_train = labels[split]

                train_sample = y_train.shape[0]
                print("training examples:", train_sample)
                test_sample = y_test.shape[0]
                print("test examples    :", test_sample)

                # define model
                with tf.device('/cpu:0'):
                    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                                  routings=args.routings,
                                                                  model_version= model_version,
                                                                  lam_regularize = args.lam_regularize)
                model.summary()
                plot_model(model, to_file=args.save_dir+'/model_fold'+str(curr_fold)+'.png', show_shapes=True)

                # define muti-gpu model
                multi_model = multi_gpu_model(model, gpus=args.gpus)

                # train
                train_start_time = time.time()
                train(model=multi_model, data=((x_train, y_train), (x_test, y_test)), args=args,fold=curr_fold)
                train_used_time_fold = time.time() - train_start_time
                model.save_weights(args.save_dir + '/trained_model_fold'+str(curr_fold)+'.h5')
                print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
                print('Train time: ', train_used_time_fold)

                #test
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  Begin: test' + '-' * 30)
                test_start_time = time.time()
                y_pred = eval_model.predict(x_test, batch_size=100)  # batch_size = 100
                test_used_time_fold = time.time() - test_start_time
                test_acc_fold = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                #print('shape of y_pred: ',y_pred.shape[0])
                #print('y_pred: ', y_pred)
                #print('y_test: ', y_test)
                print('(' + time.asctime(time.localtime(time.time())) + ') Test acc:', test_acc_fold, 'Test time: ',test_used_time_fold )
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  End: test' + '-' * 30)
                test_accuracy_allfold = np.append(test_accuracy_allfold, test_acc_fold)
                train_used_time_allfold = np.append(train_used_time_allfold, train_used_time_fold)
                test_used_time_allfold = np.append(test_used_time_allfold, test_used_time_fold)

                K.clear_session()

            summary = pd.DataFrame({'fold': range(1,fold+1), 'Test accuracy': test_accuracy_allfold, 'train time': train_used_time_allfold, 'test time': test_used_time_allfold})
            hyperparam = pd.DataFrame({'average acc of 10 folds': np.mean(test_accuracy_allfold), 'average train time of 10 folds': np.mean(train_used_time_allfold), 'average test time of 10 folds': np.mean(test_used_time_allfold),'epochs': args.epochs, 'lr':args.lr, 'batch size': args.batch_size},index=['dimention/sub'])
            writer = pd.ExcelWriter(args.save_dir + '/'+'summary'+ '_'+subject+'.xlsx')
            summary.to_excel(writer, 'Result', index=False)
            hyperparam.to_excel(writer, 'HyperParam', index=False)
            writer.save()
            print('10 fold average accuracy: ', np.mean(test_accuracy_allfold))
            print('10 fold average train time: ', np.mean(train_used_time_allfold))
            print('10 fold average test time: ', np.mean(test_used_time_allfold))


