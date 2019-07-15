#encoding:UTF-8
import numpy as np
'''
@param labels:保留哪些 标签  / which labels will be used
'''
def load_minst(labels):
    print("load minst...")
    #处理csv文件  usecols为你想操作的列  skiprows为你想跳过的行 =1为跳过第一行
    minst_train=np.loadtxt('../data/minst_train.csv', delimiter=",",dtype=np.float32)
    minst_test = np.loadtxt('../data/minst_test.csv',delimiter=",",dtype=np.float32)
    data_train = minst_train
    data_test = minst_test
    if labels != None:
        data_train = minst_train[minst_train[:,0]==labels[0]]
        data_test = minst_test[minst_test[:,0]==labels[0]]
        for i in range(1,len(labels)):
            data_train = np.concatenate((data_train,minst_train[minst_train[:,0]==labels[i]]),axis=0)
            data_test = np.concatenate((data_test,minst_test[minst_test[:,0]==labels[i]]),axis=0)
    X_train = np.array(data_train[:,1:])
    y_train =  np.array(data_train[:,0])
    X_test =  np.array(data_test[:,1:])
    y_test = np.array(data_test[:,0])
    print("data loaded...")
    return X_train,y_train,X_test,y_test


'''
X_train,y_train,X_test,y_test = load_minst([0,1])
print("shape:")
print("x_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)
'''