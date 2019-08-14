import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool
from sklearn.model_selection import KFold
'''
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''

# 用来测试少量的数据，减少计算等待时间
multi = False
# sg迭代的阈值
threshold = 50000


'''
拿训练集数据，80%的学生
计算每道题目的s失误率和g猜测率
'''

def EStep(IL,sg,n,r,k,i):
    base = 2**(k-2)
    for l in range(i*base,(i+1)*base):
        # 学生的数量
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL

def MStep(IL,n,r,k,i):
    base = 2**(k-2)
    ni,nj=n.shape
    IR = np.zeros((4, nj))
    n1 = np.ones(n.shape)
    for l in range(i*base,(i+1)*base):
        IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
        IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
        IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
    return IR
def trainDINAModel(n,Q):
    startTime = time.time()
    print('开始训练DINA模型')
    ni, nj = n.shape
    Qi, k = Q.shape
    # 计算每道题目的s失误率和g猜测率
    sg = np.zeros((nj, 2))

    #构造K矩阵，表示k个技能可以组成的技能模式矩阵
    K = np.mat(np.zeros((k, 2 ** k), dtype=int))
    for j in range(2 ** k):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[k - len(l) + i, j] = l[i]
    #r矩阵表示理论上j这道题目对于l这个模式能否做对
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1


    # 初始化每道题目的s失误率和g猜测率，
    # sg[i][0]表示第i道题目的s失误率，sg[i][1]表示第i道题目的g猜测率
    for i in range(nj):
        sg[i][0] = 0.2
        sg[i][1] = 0.2

    continueSG = True
    kk =1
    lastLX = 1
    # 计算s和g迭代的次数
    # 学生*模式数 = 学生*  题目数         题目数*技能         技能*模式数
    while continueSG == True:
        # E步，求似然矩阵
        IL = np.zeros((ni, 2 ** k))
        # 技能模式的数量
        if multi==True:
            print('multi 4 processes')
            with Pool(processes=4) as pool:
                multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, k, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IL += item
                sumIL =IL.sum(axis=1)
                LX = np.sum([i for i in map(math.log2, sumIL)])
                print('LX')
                print(LX)
                IL = (IL.T / sumIL).T
                IR = np.zeros((4, nj))
                multiple_results = [pool.apply_async(MStep, (IL,  n, r, k, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IR += item
        else:
            print('single process')
            for l in range(2 ** k):
                lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
				1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
                IL[:,l] = lll.prod(axis=1)
            sumIL = IL.sum(axis=1)
            LX = np.sum([i for i in map(math.log2, sumIL)])
            print('LX')
            print(LX)
            IL = (IL.T / sumIL).T
            #IR中的 0 1 2 3  分别表示 IO RO I1 R1
            IR = np.zeros((4,nj))
            n1 = np.ones(n.shape)
            for l in range(2 ** k):
                IR[0] += np.sum(((1-r.A[:,l])* n1).T*IL[:,l],axis=1)
                IR[1] += np.sum(((1-r.A[:,l])* n).T*IL[:,l],axis=1)
                IR[2] += np.sum((r.A[:,l]* n1).T*IL[:,l],axis=1)
                IR[3] += np.sum((r.A[:,l]* n).T*IL[:,l],axis=1)
        #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
        # if (abs(IR[1] / IR[0] - sg[:,1])<threshold).any() and (abs((IR[2]-IR[3]) / IR[2] -sg[:,0])<threshold).any():
        if abs(LX-lastLX)<threshold:
            continueSG = False

        lastLX = LX
        sg[:,1] = IR[1] / IR[0]
        sg[:,0] = (IR[2]-IR[3]) / IR[2]
        print(str(kk ) +"次迭代，"+str(ni)+"个学生，"+str(nj)+"道题目的失误率和猜测率")
        kk +=1
    endTime = time.time()
    print('DINA模型训练消耗时间：'+str(int(endTime-startTime))+'秒')
    return sg,r,K

def trainIDINAModel(n,Q):
    startTime = time.time()
    print('开始训练IDINA模型')

    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape

    # 计算每道题目的s失误率和g猜测率
    sg = np.zeros((nj, 2))
    # 计算每个学生，在2^k中技能模式下面的似然函数，暂且将似然函数等于后验概率

    k = Qj

    # 构造K矩阵，表示k个技能可以组成的技能模式矩阵
    K = np.mat(np.zeros((k, 2 ** k), dtype=int))
    for j in range(2 ** k):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[k - len(l) + i, j] = l[i]
    # r矩阵表示理论上j这道题目对于l这个模式能否做对
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1

    # 初始化每道题目的s失误率和g猜测率，
    # sg[i][0]表示第i道题目的s失误率，sg[i][1]表示第i道题目的g猜测率
    for i in range(nj):
        sg[i][0] = 0.01
        sg[i][1] = 0.01

    continueSG = True
    kk =1
    # 计算s和g迭代的次数
    IL = np.ones((ni, 2 ** Qj))
    istart = 0
    istop = ni
    while continueSG == True:
        # E步，求似然矩阵

        for i in range(istart,istop):
            IL[i] = 1
            lll = ((1 - sg[:, 0]) ** n[i] * sg[:, 0] ** (1 - n[i])) ** r.T.A * (sg[:, 1] ** n[i] * (
            1 - sg[:, 1]) ** (1 - n[i])) ** (1 - r.T.A)
            IL[i] = lll.prod(axis=1)

        print("IL是训练集学生，所有技能模式的似然概率矩阵")
        print(IL)

        istart = istop % ni
        istop = istart + 10
        if istop > ni:
            istop = ni

            # M步，求s，g
            # 根据这四个参数来更新迭代s和g
        I0 = np.zeros(nj)
        R0 = np.zeros(nj)
        I1 = np.zeros(nj)
        R1 = np.zeros(nj)
        # l表示每一种技能模式
        n1 = np.ones(n.shape)
        for l in range(2 ** Qj):
            I1 += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
            R1 += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
            I0 += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
            R0 += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        # 针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
        if (abs(R0 / I0 - sg[:, 1]) < threshold).any() and (abs((I1 - R1) / I1 - sg[:, 0]) < threshold).any():
            continueSG = False
        sg[:, 1] = R0 / I0
        sg[:, 0] = (I1 - R1) / I1
        print(sg)
        print(str(kk) + "次迭代，" + str(ni) + "个学生，" + str(nj) + "道题目的失误率和猜测率")
        kk += 1
    endTime = time.time()
    print('IDINA模型训练消耗时间：'+str(int(endTime-startTime))+'秒')
    return sg,r

def continuously(IL):
    ni,nj = IL.shape
    Qj = (int)(math.log2(nj))
    continuous = np.ones((ni, Qj))
    denominator = np.sum(IL, axis=1)
    for j in range(Qj):
        molecule = np.zeros(ni)
        for l in range(nj):
            ll = list(bin(l).replace('0b', ''))
            if j < len(ll) and ll[len(ll) - j - 1] == '1':
                molecule += IL[:, l]
        continuous[:, Qj - 1 - j] = molecule / denominator
    return continuous
def discrete(continuous):
    ni,k = continuous.shape
    a = np.zeros(ni,dtype=int)
    for i in range(ni):
        for ki in range(k):
            if continuous[i][ki]>0.5:
                a[i] += 2**(k-ki-1)
    return a
def predictDINA(n,Q,sg,r,K):
    startTime = time.time()
    print('预测开始')

    ni, nj = n.shape
    Qi, Qj = Q.shape
    # 预测的每个学生的技能向量
    IL = np.zeros((ni, 2 ** Qj))
    k = Qj

    if multi == True:
        print('预测 multi 4 processes')
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, k, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item
    else:
        for l in range(2 ** Qj):
            # 学生的数量
            lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
            IL[:, l] = lll.prod(axis=1)
    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)

    # a2 = discrete(continuously(IL))

    # print('连续化向量')
    # print(continuous)
    # 计算准确率

    i, j = n.shape
    print('总共有' + str(ni) + '个人，a准确率为：')
    # print(K)
    pd.DataFrame(K[:,a].T).to_csv("math2_skill_2500.csv",index=False)
    p1 = np.sum((r[:, a] == n.T) * 1) / (i * j)
    print(p1)
    # print('总共有' + str(ni) + '个人，a2准确率为：')
    # p1 = np.sum((r[:, a2] == n.T) * 1) / (i * j)
    # print(p1)
    print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    print('-----------------------------------------------')
    return p1


def testPredict(model,dataSet):
    print('dataSet is',dataSet)
    if dataSet == 'FrcSub':
        n = pd.read_csv('math2015/FrcSub/data.csv').values
        Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    elif dataSet == 'Math1':
        n = pd.read_csv('math2015/Math1/data.csv').values
        Q = np.mat(pd.read_csv('math2015/Math1/q.csv').head(15).values)
    elif dataSet == 'Math2':
        n = pd.read_csv('math2015/Math2/data.csv').head(2500).values
        Q = np.mat(pd.read_csv('math2015/Math2/q.csv').head(16).values)
    else:
        print('no dataSet exist!')
        exit(0)
    KF = KFold(n_splits=5,shuffle=True)
    precision = 0
    sg,r,K = trainDINAModel(n, Q)
    pd.DataFrame(sg).to_csv("csvFile/%s_sg_%s.csv" %(dataSet,model), index=False)
    predictDINA(n, Q, sg, r,K)
    # for train_index, test_index in KF.split(n):
    #     X_train, X_test = n[train_index], n[test_index]
    #     if model == 'DINA':
    #         sg,r = trainDINAModel(X_train,Q)
    #     else:
    #         sg,r = trainIDINAModel(X_train,Q)
    #     precision += predictDINA(X_test, Q, sg, r)

    print('准确率平均值')
    print(precision/5)

def main():
    print('main starting')
    startTime = time.time()
    dataSet = ['FrcSub', 'Math1', 'Math2']
    testPredict('DINA', dataSet[0])
    print('总时间:', time.time() - startTime)
if __name__ == "__main__":
    main()


