import numpy as np
import pandas as pd
import time
from operator import mul
from functools import reduce
'''
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''

# 用来测试少量的数据，减少计算等待时间
headNum = 1000
# sg迭代的阈值
threshold = 0.001


'''
拿训练集数据，80%的学生
计算每道题目的s失误率和g猜测率
'''

def trainDINAModel(dataSet):
    startTime = time.time()
    print('开始训练DINA模型')

    # 可以使用head函数控制读取的学生数量
    # sample表示采样，frac=0.8表示随机采样80%的记录
    if dataSet =='FrcSub':
        # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
        n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.8).values
        # Q 表示每道题目答对需要掌握的知识点向量
        Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    else :
        n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.8).values
        Q = np.mat(pd.read_csv('math2015/Math1/q.csv').head(15).values)
    # 536*0.8个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape

    k = Qj
    # 计算每道题目的s失误率和g猜测率
    sg = np.zeros((nj, 2))
    # 计算每个学生，在2^k中技能模式下面的似然函数，暂且将似然函数等于后验概率


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
        sg[i][0] = 0.01
        sg[i][1] = 0.01

    continueSG = True
    kk =1
    # 计算s和g迭代的次数
    # 学生*模式数 = 学生*  题目数         题目数*技能         技能*模式数
    while continueSG == True:
        # E步，求似然矩阵
        IL = np.ones((ni, 2 ** Qj))
        # 技能模式的数量
        for l in range(2 ** Qj):
            # 学生的数量
            for i in range(ni):
                # 将全部题目的似然概率相乘  得到i学生对于l模式的后验概率，
                # r.T[l].tolist()[0]表示理论情况与n[i]实际情况结合选择该道题目的似然概率
                ll = map(lambda nl, nk, s, g: ((1 - s)**nk*s**(1-nk))**nl *(g**nk*(1 - g)**(1-nk))**(1-nl),
                        n[i] , r.T[l].tolist()[0], sg[:,0], sg[:,1])
                IL[i][l] = reduce(mul, ll)
        print("IL是训练集学生，所有技能模式的似然概率矩阵")
        print(IL)

        # M步，求s，g
        # 根据这四个参数来更新迭代s和g
        I0 = np.zeros(nj)
        R0 = np.zeros(nj)
        I1 = np.zeros(nj)
        R1 = np.zeros(nj)
        # l表示每一种技能模式
        n1 = np.ones(n.shape)
        for l in range(2 ** Qj):
            I1 += np.sum((r.A[:,l]* n1).T*IL[:,l],axis=1)
            R1 += np.sum((r.A[:,l]* n).T*IL[:,l],axis=1)
            I0 += np.sum(((1-r.A[:,l])* n1).T*IL[:,l],axis=1)
            R0 += np.sum(((1-r.A[:,l])* n).T*IL[:,l],axis=1)
        #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
        if (abs(R0 / I0 - sg[:,1])<threshold).any() and (abs((I1-R1) / I1 -sg[:,0])<threshold).any():
            continueSG = False
        sg[:,1] = R0 / I0
        sg[:,0] = (I1-R1) / I1
        print(sg)
        print(str(kk ) +"次迭代，"+str(ni)+"个学生，"+str(nj)+"道题目的失误率和猜测率")
        kk +=1
    endTime = time.time()
    print('DINA模型训练消耗时间：'+str(int(endTime-startTime))+'秒')
    return sg,r

def trainIDINAModel(dataSet):
    startTime = time.time()
    print('开始训练IDINA模型')

    # 可以使用head函数控制读取的学生数量
    # sample表示采样，frac=0.8表示随机采样80%的记录
    if dataSet =='FrcSub':
        # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
        n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.8).values
        # Q 表示每道题目答对需要掌握的知识点向量
        Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    else :
        n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.8).values
        Q = np.mat(pd.read_csv('math2015/Math1/q.csv').head(15).values)

    # 536*0.8个学生,20道题目
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

        # 技能模式的数量
        for l in range(2 ** Qj):
            # 学生的数量
            for i in range(istart,istop):
                # 题目的数量
                IL[i][l] = 1
                ll = map(lambda nl, nk, s, g: ((1 - s) ** nk * s ** (1 - nk)) ** nl * (g ** nk * (1 - g) ** (1 - nk)) ** (1 - nl),n[i], r.T[l].tolist()[0], sg[:, 0], sg[:, 1])
                IL[i][l] = reduce(mul, ll)

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

def testPredict(model,dataSet):
    # 得到模型的参数，s和g,
    if model == 'DINA':
        sg,r = trainDINAModel(dataSet)
    else:
        sg,r = trainIDINAModel(dataSet)
    startTime = time.time()
    print('预测开始')
    # 测试集随机挑选20%的数据
    if dataSet == 'FrcSub':
        n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.2).values
        Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    else:
        n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.2).values
        Q = np.mat(pd.read_csv('math2015/Math1/q.csv').head(15).values)
    # 536*0.2个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape
    # 预测的每个学生的技能向量
    IL = np.ones((ni, 2 ** Qj))

    for l in range(2 ** Qj):
        # 学生的数量
        for i in range(ni):
            # 题目的数量
            ll = map(lambda nl, nk, s, g: ((1 - s) ** nk * s ** (1 - nk)) ** nl * (g ** nk * (1 - g) ** (1 - nk)) ** (1 - nl), n[i], r.T[l].tolist()[0], sg[:, 0], sg[:, 1])
            IL[i][l] = reduce(mul, ll)
    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)
    print('测试集中各个学生的预测的a技能向量')
    print(a)

    continuous = np.ones((ni,Qj))
    denominator = np.sum(IL,axis=1)
    for j in range(Qj):
        molecule = np.zeros(ni)
        for l in range(2**Qj):
            ll = list(bin(l).replace('0b', ''))
            if j<len(ll) and ll[len(ll)-j-1]=='1':
                molecule += IL[:,l]
        continuous[:,Qj-1-j] = molecule/denominator

    print('连续化向量')
    # print(continuous)


    # 计算准确率
    i, j = n.shape
    print('总共有'+str(ni)+'个人，准确率为：')
    print(np.sum((r[:, a] == n.T) * 1) / (i * j))
    print('预测消耗时间：' + str(int(time.time())-int(startTime))+'秒')

def main():
    testPredict('IDINA','FrcSub1')

if __name__ == "__main__":
    main()
