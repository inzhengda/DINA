import numpy as np
import pandas as pd
import time
import math
from operator import mul
from functools import reduce
'''
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''

# 用来测试少量的数据，减少计算等待时间
headNum = 536
# sg迭代的阈值
threshold = 0.001




'''
判别qk这道题目在传进来的l技能模式下能否做对题目

参数表示：
qk表示q矩阵里面第k道题目需要满足的技能向量
l表示技能模式，是int整数，范围在0~2^k-1

将q向量转为2进制的字符串，再转为int整数，按位与之后如果还是等于q自身，
说明q要求的技能向量，l这种模式都满足，即l能做对qk这道题目
'''


def nTrueOrFalse(qk, l):
    q =int(''.join(map(str, qk)), 2)
    if q&l == q:
        return 1
    else:
        return 0



'''
拿训练集数据，80%的学生
计算每道题目的s失误率和g猜测率
'''

def trainDINAModel():
    startTime = time.time()
    print('训练模型开始时间：'+str(int(startTime)))

    # 可以使用head函数控制读取的学生数量
    # sample表示采样，frac=0.8表示随机采样80%的记录
    # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
    n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.8).values
    # n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.8).values
    # Q 表示每道题目答对需要掌握的知识点向量
    # Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
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
        for l in range(2 ** Qj):
            # i表示每一个学生
            for i in range(ni):
                # I1,R1的情况，表示理论上能做对

                I1 += r.A[:,l]* IL[i][l]
                R1 += r.A[:,l]* IL[i][l]*n[i]
                I0 += (1-r.A[:,l])* IL[i][l]
                R0 += (1-r.A[:,l])* IL[i][l]*n[i]

        #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
        # if (abs(R0 / I0 - sg[:,1])<threshold).any() and (abs((I1-R1) / I1 -sg[:,0])<threshold).any():
        if kk ==2:
            continueSG = False
        sg[:,1] = R0 / I0
        sg[:,0] = (I1-R1) / I1
        print(sg)
        print(str(kk ) +"次迭代，"+str(ni)+"个学生，"+str(nj)+"道题目的失误率和猜测率")
        kk +=1
    endTime = time.time()
    print('训练消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg,r

def trainIDINAModel():
    startTime = time.time()
    print('训练模型开始时间：'+str(int(startTime)))

    # 可以使用head函数控制读取的学生数量
    # sample表示采样，frac=0.8表示随机采样80%的记录
    # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
    # n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.8)
    n = pd.read_csv('math2015/FrcSub/data.csv').head(headNum).sample(frac=0.8)
    # Q 表示每道题目答对需要掌握的知识点向量
    Q = pd.read_csv('math2015/FrcSub/q.csv')

    # 536*0.8个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape

    # 计算每道题目的s失误率和g猜测率
    sg = np.zeros((nj, 2))
    # 计算每个学生，在2^k中技能模式下面的似然函数，暂且将似然函数等于后验概率


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
                for j in range(nj):
                    # 计算Q矩阵中，第k题的情况下，l这个技能模式能否答对
                    nl = nTrueOrFalse((Q.iloc[j]), l)
                    # 理论上，答对的情况
                    if nl == 1:
                        # 实际该学生，对于k这道题做对了，似然函数*(1-s)
                        if n.iloc[i][j] == 1:
                            IL[i][l] *= (1 - sg[j][0])
                        # 实际该学生，对于k这道题居然做错了，似然函数*s，失误率
                        else:
                            IL[i][l] *= sg[j][0]
                    # 理论上，答错的情况
                    else:
                        # 实际该学生，对于k这道题居然做对了，似然函数*g，猜测率
                        if n.iloc[i][j] == 1:
                            IL[i][l] *= sg[j][1]
                        # 实际该学生，对于k这道题确实做错了，似然函数*(1-g)
                        else:
                            IL[i][l] *= 1 - sg[j][1]

            print('l:' + str(l))
        print("IL是训练集学生，所有技能模式的似然概率矩阵")
        print(IL)

        istart = istop % ni
        istop = istart + 10
        if istop > ni:
            istop = ni

        # M步，求s，g
        # k表示每一道题目
        for j in range(nj):
            # 根据这四个参数来更新迭代s和g
            I0 = 0
            R0 = 0
            I1 = 0
            R1 = 0
            #l表示每一种技能模式
            for l in range(2 ** Qj):
                #i表示每一个学生
                for i in range(ni):
                    #也需要计算k这道题，l这模式能否做对
                    nl = nTrueOrFalse((Q.iloc[j]),l)
                    # I1,R1的情况，表示理论上能做对
                    if nl ==1:
                        I1 += IL[i][l]
                        # 实际也做对的情况
                        if n.iloc[i][j] ==1:
                            R1 += IL[i][l]
                    # I0,R0的情况，表示理论上不能做对
                    else:
                        I0 += IL[i][l]
                        # 实际却做对的情况
                        if n.iloc[i][j] ==1:
                            R0 += IL[i][l]

            #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
            if abs(R0 / I0 - sg[j][1])<threshold and abs((I1-R1) / I1 -sg[j][0])<threshold:
                continueSG = False
            sg[j][1] = R0 / I0
            sg[j][0] = (I1-R1) / I1
            print('-------------------------------')
            print(str(j + 1) + '题目finish')

        print(sg)
        print(str(kk ) +"次迭代，"+str(ni)+"个学生，20道题目的失误率和猜测率")
        kk +=1
    endTime = time.time()
    print('********************************************')
    print('训练消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg



def testPredict():
    # 得到模型的参数，s和g,
    sg,r = trainDINAModel()
    # sg = trainDINAModel()
    startTime = time.time()
    print('预测开始时间：' + str(int(startTime)))
    # 测试集随机挑选20%的数据
    n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.2).values
    # n = pd.read_csv('math2015/FrcSub/data.csv').sample(frac=0.2).values
    # Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
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

    # 计算准确率
    i, j = n.shape
    print('总共有'+str(ni)+'个人，准确率为：')
    print(np.sum((r[:, a] == n.T) * 1) / (i * j))
    print('预测消耗时间：' + str(int(time.time())-int(startTime))+'秒')

def main():
    testPredict()



if __name__ == "__main__":
    main()
