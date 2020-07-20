import xlrd
import pandas as pd
import numpy as np

subjects = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16',
            '17','18','19','20','21','22','23']
dataset_name = 'dreamer'
model_version = 'v2'
epoch = '30' #v0 20; v2 40
debase = 'no'

summary_valence = pd.DataFrame()
summary_arousal = pd.DataFrame()
summary_dominance = pd.DataFrame()

for label in ['valence']: # ,'arousal','dominance']:#['valence','arousal','dominance']:
    test_accuracy_allsub = np.zeros(shape=[0], dtype=float)
    test_time_allsub = np.zeros(shape=[0], dtype=float)
    train_time_allsub = np.zeros(shape=[0], dtype=float)
    epoch_allsub = np.zeros(shape=[0], dtype=float)
    lr_allsub = np.zeros(shape=[0], dtype=float)
    batch_size_allsub = np.zeros(shape=[0], dtype=float)
    for sub in subjects:
        print(sub)
        xl = xlrd.open_workbook(r'/home/bsipl_5/experiment/MLF-CapsNet/result_'+dataset_name + '_redo'+'/sub_dependent_'+model_version+'/'+debase+'/'+sub+'_'+label+epoch+'/summary_'+sub+'.xlsx')
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
    elif label == 'arousal':
        summary_arousal = summary
    else:
        summary_dominance = summary

writer = pd.ExcelWriter('/home/bsipl_5/experiment/MLF-CapsNet/result_'+dataset_name+'_redo'+'/'+dataset_name+'_'+model_version+'_'+debase+'.xlsx')
summary_valence.to_excel(writer, 'valence', index=False)
summary_arousal.to_excel(writer, 'arousal', index=False)
summary_dominance.to_excel(writer, 'dominance', index=False)
writer.save()


