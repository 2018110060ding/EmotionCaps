import xlrd
import pandas as pd
import numpy as np

subjects = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16',
            's17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28','s29','s30','s31','s32'] # For Dominance, the 's27' should be removed.
dataset_name = 'deap' # the name of dataset: deap or dreamer
model_version = 'v1' # the version of model
epoch = '40' # v0 20; v2 40
debase = 'yes' # with or without debaseline

summary_valence = pd.DataFrame()
summary_arousal = pd.DataFrame()
summary_dominance = pd.DataFrame()

for label in ['valence','arousal']:
    test_accuracy_allsub = np.zeros(shape=[0], dtype=float)
    test_time_allsub = np.zeros(shape=[0], dtype=float)
    train_time_allsub = np.zeros(shape=[0], dtype=float)
    epoch_allsub = np.zeros(shape=[0], dtype=float)
    lr_allsub = np.zeros(shape=[0], dtype=float)
    batch_size_allsub = np.zeros(shape=[0], dtype=float)
    for sub in subjects:
        print(sub)
        xl = xlrd.open_workbook(r'/home/bsipl_5/experiment/MLF-CapsNet/result_'+dataset_name+'/sub_dependent_'+model_version+'/'+debase+'/'+sub+'_'+label+epoch+'/summary_'+sub+'.xlsx')
        table = xl.sheets()[1]
        acc = table.cell(1,0).value
        test_time = table.cell(1,1).value
        train_time = table.cell(1, 2).value
        batch_size = table.cell(1, 3).value
        epochs = table.cell(1, 4).value
        lr = table.cell(1, 5).value

        test_accuracy_allsub = np.append(test_accuracy_allsub, acc)
        test_time_allsub = np.append(test_time_allsub, test_time)
        train_time_allsub =np.append(train_time_allsub,train_time)
        batch_size_allsub = np.append(batch_size_allsub,batch_size)
        epoch_allsub =np.append(epoch_allsub,epochs)
        lr_allsub = np.append(lr_allsub,lr)

    summary = pd.DataFrame(
        {'Subjects': subjects, 'average acc of 10 folds': test_accuracy_allsub, 'average train time of 10 folds': train_time_allsub,
         'average test time of 10 folds': test_time_allsub, 'epochs': epoch_allsub, 'batch size': batch_size_allsub,'lr':lr_allsub})
    if label == 'valence':
        summary_valence = summary
    elif label =='arousal':
        summary_arousal = summary
    elif label == 'dominance':
        summary_dominance = summary

writer = pd.ExcelWriter('/home/bsipl_5/experiment/MLF-CapsNet/result_'+dataset_name+'/'+dataset_name+'_'+model_version+'_'+debase+'.xlsx')
summary_valence.to_excel(writer, 'valence', index=False)
summary_arousal.to_excel(writer, 'arousal', index=False)
summary_dominance.to_excel(writer, 'dominance', index=False)
writer.save()


