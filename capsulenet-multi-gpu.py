"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet-multi-gpu.py
       python capsulenet-multi-gpu.py --gpus 2
       ... ...

Result:
    About 55 seconds per epoch on two GTX1080Ti GPU cards

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import optimizers
from keras import backend as K

K.set_image_data_format('channels_last')

from capsulenet import CapsNet, margin_loss, manipulate_latent ,test
from deap_preprocess01 import deap_preprocess
import pandas as pd
import time

time_start_whole = time.time()

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
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}_fold'+str(fold)+'.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (1.0 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    """
    # Training with validation set
    model.fit([x_train, y_train],[y_train, x_train],  batch_size=args.batch_size, epochs=args.epochs,verbose = 1,
              validation_split= 0.1 , callbacks=[log, tb, checkpoint, lr_decay])
    """
    # Training without validation set
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs, verbose=1,
               callbacks=[log, tb, lr_decay])
    """
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint,lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    """

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model

"""
def test(model, data, fold):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)  # batch_size = 100
    print('-'*30 + 'fold  ' + fold+ 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
"""

subjects = ['s30']  #'s20','s21','s22','s23','s24','s25','s26','s27','s28','s29','s30','s31','s32']
dimentions = ['valence']

for subject in subjects:
    time_start_onesub = time.time()
    for dimention in dimentions:
        time_start_onedimention = time.time()
        if __name__ == "__main__":
            import numpy as np
            import tensorflow as tf
            import os
            from keras.preprocessing.image import ImageDataGenerator
            from keras import callbacks
            from keras.utils.vis_utils import plot_model
            from keras.utils import multi_gpu_model

            # setting the hyper parameters
            import argparse
            parser = argparse.ArgumentParser(description="Capsule Network on DEAP.")
            parser.add_argument('--epochs', default=20, type=int)
            parser.add_argument('--batch_size', default=100, type=int)
            parser.add_argument('--lam_recon', default=0.0, type=float,
                                help="The coefficient for the loss of decoder")
            parser.add_argument('-r', '--routings', default=3, type=int,
                                help="Number of iterations used in routing algorithm. should > 0")
            parser.add_argument('--shift_fraction', default=0.1, type=float,
                                help="Fraction of pixels to shift at most in each direction.")
            parser.add_argument('--debug', default=0, type=int,
                                help="Save weights by TensorBoard")
            #parser.add_argument('--save_dir', default='./result') #result of training with validation set
            #parser.add_argument('--save_dir', default='./result_without_val')  # result of training without validation set
            parser.add_argument('--save_dir', default='./result_other') # other
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('--digit', default=5, type=int,
                                help="Digit to manipulate")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            parser.add_argument('--lr', default=0.0001, type=float,
                                help="Initial learning rate")
            parser.add_argument('--gpus', default=2, type=int)
            args = parser.parse_args()

            print(time.asctime(time.localtime(time.time())))
            print(args)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

            # load data
            #(x_train, y_train), (x_test, y_test) = load_deapdata()
            #subject = 's04'
            #dimention = 'valence'
            datasets,labels = deap_preprocess(subject,dimention)

            args.save_dir = args.save_dir + '/' + subject + '_' + dimention
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            fold = 10
            test_accuracy = np.zeros(shape=[0], dtype=float)
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
                                                                  routings=args.routings)#, manipulate_model
                model.summary()
                plot_model(model, to_file=args.save_dir+'/model_fold'+str(curr_fold)+'.png', show_shapes=True)

                # train or test
                if args.weights is not None:  # init the model weights with provided one
                    model.load_weights(args.weights)
                if not args.testing:
                    # define muti-gpu model
                    multi_model = multi_gpu_model(model, gpus=args.gpus)
                    train(model=multi_model, data=((x_train, y_train), (x_test, y_test)), args=args,fold=curr_fold)
                    model.save_weights(args.save_dir + '/trained_model_fold'+str(curr_fold)+'.h5')
                    print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
                    #test(model=eval_model, data=(x_test, y_test), args=args)

                    #test
                    print('-' * 30 + 'fold  ' + str(curr_fold) + '  Begin: test' + '-' * 30)
                    y_pred, x_recon = eval_model.predict(x_test, batch_size=100)  # batch_size = 100
                    test_acc_fold = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                    print('shape of y_pred: ',y_pred.shape[0])
                    print('last 40 of y_pred: ', y_pred[200:240,:])
                    print('last 40 of y_test: ', y_test[200:240,:])
                    print('(' + time.asctime(time.localtime(time.time())) + ') Test acc:', test_acc_fold)
                    print('-' * 30 + 'fold  ' + str(curr_fold) + '  End: test' + '-' * 30)
                    test_accuracy = np.append(test_accuracy, test_acc_fold)

                else:  # as long as weights are given, will run testing
                    if args.weights is None:
                        print('No weights are provided. Will test using random initialized weights.')
                    #manipulate_latent(manipulate_model, (x_test, y_test), args)
                    test(model=eval_model, data=(x_test, y_test), args=args)
            time_used_onedimention = time.time()-time_start_onedimention
            acc_summary = pd.DataFrame({'fold': range(1,fold+1), 'accuracy': test_accuracy})
            time_used = pd.DataFrame({'Time used': time_used_onedimention},index=['dimention/sub'])
            writer = pd.ExcelWriter(args.save_dir + '/'+"summary.xlsx")
            acc_summary.to_excel(writer, 'accuracy', index=False)
            time_used.to_excel(writer,'Time used',index=False)
            writer.save()
    print('Time used of one subject: ', time.time() - time_start_onesub)

print('Time used of all subjects: ',time.time() - time_start_whole)