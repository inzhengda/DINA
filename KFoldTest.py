from sklearn.model_selection import KFold
import numpy as np
X=np.array([[1,2],[3,4],[1,3],[3,5]])
Y=np.array([1,2,3,4])
KF=KFold(n_splits=2)  #建立4折交叉验证方法  查一下KFold函数的参数
for train_index,test_index in KF.split(X):
    print("TRAIN:",train_index,"TEST:",test_index)
    X_train,X_test=X[train_index],X[test_index]
    Y_train,Y_test=Y[train_index],Y[test_index]
    print(X_train,X_test)
    print(Y_train,Y_test)