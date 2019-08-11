import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool
'''
学生素质综合诊断结果
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''

# 用来测试少量的数据，减少计算等待时间
multi = False
# sg迭代的阈值
threshold = 10


'''
拿训练集数据，80%的学生
计算每道题目的s失误率和g猜测率
'''

def EStep(IL,sg,n,r,k,i):
    base = 2**(k-2)
    for l in range(i*base,(i+1)*base):
        # 学生的数量
        lll = n*((1 - sg[:, 0]) ** r.T.A[l] * sg[:, 1] ** (1 - r.T.A[l])) + (1 - n) * ((sg[:, 0]) ** r.T.A[l] * (1-sg[:, 1]) ** (1 - r.T.A[l]))
        print(lll)
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
    print(r)

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
                # 学生的数量
                lll = n * ((1 - sg[:, 0]) ** r.T.A[l] * sg[:, 1] ** (1 - r.T.A[l])) + (1 - n) * (
                (sg[:, 0]) ** r.T.A[l] * (1 - sg[:, 1]) ** (1 - r.T.A[l]))
                # print(lll)
                IL[:,l] = lll.prod(axis=1)
            sumIL = IL.sum(axis=1)

            IL = (IL.T / sumIL).T
            # print(IL)
            # print(sumIL)
            LX = np.sum([i for i in map(math.log2, sumIL)])

            print('LX')
            print(LX)
            # IL = (IL.T / sumIL).T
            #IR中的 0 1 2 3  分别表示 IO RO I1 R1
            IR = np.zeros((4,nj))
            n1 = np.ones(n.shape)
            n2 = (n>0.4)*1
            for l in range(2 ** k):
                IR[0] += np.sum(((1-r.A[:,l])* n1).T*IL[:,l],axis=1)
                IR[1] += np.sum(((1-r.A[:,l])* n2).T*IL[:,l],axis=1)
                IR[2] += np.sum((r.A[:,l]* n1).T*IL[:,l],axis=1)
                IR[3] += np.sum((r.A[:,l]* n2).T*IL[:,l],axis=1)
        #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
        # if (abs(IR[1] / IR[0] - sg[:,1])<threshold).any() and (abs((IR[2]-IR[3]) / IR[2] -sg[:,0])<threshold).any():
        if abs(LX-lastLX)<threshold:
            continueSG = False
        # print(sg)
        lastLX = LX
        sg[:,1] = IR[1] / IR[0]
        sg[:,0] = (IR[2]-IR[3]) / IR[2]
        print(str(kk ) +"次迭代，"+str(ni)+"个学生，"+str(nj)+"道题目的失误率和猜测率")
        kk +=1
    endTime = time.time()
    print('DINA模型训练消耗时间：'+str(int(endTime-startTime))+'秒')
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
def predictDINA(n,Q,sg,r):
    startTime = time.time()
    print('预测开始')
    print(sg)
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
            lll = n * ((1 - sg[:, 0]) ** r.T.A[l] * sg[:, 1] ** (1 - r.T.A[l])) + (1 - n) * (
                (sg[:, 0]) ** r.T.A[l] * (1 - sg[:, 1]) ** (1 - r.T.A[l]))
            IL[:, l] = lll.prod(axis=1)
    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)

    # a2 = discrete(continuously(IL))
    # print(IL)
    continuous =continuously(IL)

    # tc = (continuous>0.5)*1
    #
    # k = 1
    #
    #
    # #0 - 1
    # print(np.sum(n[:, 0])-2*np.sum(n[:,0]*tc[:,k]))
    # print('连续化向量')
    # print(continuous.shape)
    print(continuous)
    std = np.sum(Q, axis=1)
    # pn为预测的的值
    pn = continuous * Q.T/std.T
    # print(pn)
    showLD(continuous[100])
    # 计算准确率

    i, j = n.shape
    # print('总共有' + str(ni) + '个人，a准确率为：')
    p1 = np.sum((abs(pn- n)<=0.4) * 1) / (i * j)
    # print(p1)
    # print('总共有' + str(ni) + '个人，a2准确率为：')
    # p1 = np.sum((r[:, a2] == n.T) * 1) / (i * j)
    # print(p1)
    # print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    # print('-----------------------------------------------')
    return p1

from sklearn.model_selection import KFold
def testPredict():
    n = pd.read_csv('123.csv').values
    Q = np.mat(pd.read_csv('q111.csv'))
    # n = pd.read_csv('math2015/FrcSub/data.csv').values
    # Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    KF = KFold(n_splits=5,shuffle=True)
    precision = 0
    for train_index, test_index in KF.split(n):
        X_train, X_test = n[train_index], n[test_index]
        sg,r = trainDINAModel(X_train,Q)
        print('************************************************')
        precision += predictDINA(X_test, Q, sg, r)
    print('准确率平均值')
    print(precision/5)
def main():
    startTime = time.time()
    testPredict()
    print('总时间:')
    print(time.time()-startTime)
def showLD(data):
    import matplotlib.pyplot as plt
    labels = np.array(['写作能力','良好价值观','组织能力','团队协作','演讲能力','PPT制作','表达能力','出勤率高'])
    # 数据个数
    dataLenth = 8
    # 数据
    data = np.array([0.75, 0.92, 0.73, 0.87, 0.93, 0.65, 0.78, 0.97])

    angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, 'ro-', linewidth=2)
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    ax.set_title("学生素质综合诊断结果", va='bottom', fontproperties="SimHei")
    ax.grid(True)
    plt.show()
if __name__ == "__main__":
    main()
