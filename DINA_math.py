import numpy as np
import pandas as pd
import time
import math
'''
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''

# 用来测试少量的数据，减少计算等待时间
headNum = 10



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
    n = pd.read_csv('math2015/Math1/data.csv').sample(frac=0.8)
    # n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.8)
    # Q 表示每道题目答对需要掌握的知识点向量
    Q = pd.read_csv('math2015/Math1/q.csv')

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
    while continueSG == True:
        # E步，求似然矩阵
        IL = np.ones((ni, 2 ** Qj))
        # 技能模式的数量
        print('l总数:'+str(2 ** Qj)+'i总数:'+str(ni))
        for l in range(2 ** Qj):
            # 学生的数量
            for i in range(ni):
                # 题目的数量
                for k in range(nj):
                    # 计算Q矩阵中，第k题的情况下，l这个技能模式能否答对
                    nl = nTrueOrFalse((Q.iloc[k]), l)
                    # 理论上，答对的情况
                    if nl == 1:
                        # 实际该学生，对于k这道题做对了，似然函数*(1-s)
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= (1 - sg[k][0])
                        # 实际该学生，对于k这道题居然做错了，似然函数*s，失误率
                        else:
                            IL[i][l] *= sg[k][0]
                    # 理论上，答错的情况
                    else:
                        # 实际该学生，对于k这道题居然做对了，似然函数*g，猜测率
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= sg[k][1]
                        # 实际该学生，对于k这道题确实做错了，似然函数*(1-g)
                        else:
                            IL[i][l] *= 1 - sg[k][1]
            print('l:'+str(l))
        print("IL是训练集学生，所有技能模式的似然概率矩阵")
        print(IL)



        # M步，求s，g
        # k表示每一道题目
        for k in range(nj):
            # 根据这四个参数来更新迭代s和g
            I0 = 0
            R0 = 0
            I1 = 0
            R1 = 0
            # l表示每一种技能模式
            for l in range(2 ** Qj):

                # 也需要计算k这道题，l这模式能否做对
                nl = nTrueOrFalse((Q.iloc[k]), l)
                # i表示每一个学生
                for i in range(ni):
                    # I1,R1的情况，表示理论上能做对
                    if nl ==1:
                        I1 += IL[i][l]
                        # 实际也做对的情况
                        if n.iloc[i][k] ==1:
                            R1 += IL[i][l]
                    # I0,R0的情况，表示理论上不能做对
                    else:
                        I0 += IL[i][l]
                        # 实际却做对的情况
                        if n.iloc[i][k] ==1:
                            R0 += IL[i][l]

            #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
            if abs(R0 / I0 - sg[k][1])<0.00001 and abs((I1-R1) / I1 -sg[k][0]<0.00001):
                continueSG = False
            sg[k][1] = R0 / I0
            sg[k][0] = (I1-R1) / I1
            print('-------------------------------')
            print(str(k + 1) + '题目finish')

        print(str(kk ) +"次迭代，"+str(ni)+"个学生，20道题目的失误率和猜测率")
        print(sg)
        kk +=1
    endTime = time.time()
    print('********************************************')
    print('训练消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg


def trainIDINAModel():
    startTime = time.time()
    print('训练模型开始时间：'+str(int(startTime)))

    # 可以使用head函数控制读取的学生数量
    # sample表示采样，frac=0.8表示随机采样80%的记录
    # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
    n = pd.read_csv('math2015/Math1/data.csv').sample(frac=0.8)
    # n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.8)
    # Q 表示每道题目答对需要掌握的知识点向量
    Q = pd.read_csv('math2015/Math1/q.csv')

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
                for k in range(nj):
                    # 计算Q矩阵中，第k题的情况下，l这个技能模式能否答对
                    nl = nTrueOrFalse((Q.iloc[k]), l)
                    # 理论上，答对的情况
                    if nl == 1:
                        # 实际该学生，对于k这道题做对了，似然函数*(1-s)
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= (1 - sg[k][0])
                        # 实际该学生，对于k这道题居然做错了，似然函数*s，失误率
                        else:
                            IL[i][l] *= sg[k][0]
                    # 理论上，答错的情况
                    else:
                        # 实际该学生，对于k这道题居然做对了，似然函数*g，猜测率
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= sg[k][1]
                        # 实际该学生，对于k这道题确实做错了，似然函数*(1-g)
                        else:
                            IL[i][l] *= 1 - sg[k][1]

            print('l:' + str(l))
        print("IL是训练集学生，所有技能模式的似然概率矩阵")
        print(IL)

        istart = istop % ni
        istop = istart + 10
        if istop > ni:
            istop = ni

        # M步，求s，g
        # k表示每一道题目
        for k in range(nj):
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
                    nl = nTrueOrFalse((Q.iloc[k]),l)
                    # I1,R1的情况，表示理论上能做对
                    if nl ==1:
                        I1 += IL[i][l]
                        # 实际也做对的情况
                        if n.iloc[i][k] ==1:
                            R1 += IL[i][l]
                    # I0,R0的情况，表示理论上不能做对
                    else:
                        I0 += IL[i][l]
                        # 实际却做对的情况
                        if n.iloc[i][k] ==1:
                            R0 += IL[i][l]

            #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
            if abs(R0 / I0 - sg[k][1])<0.00001 and abs((I1-R1) / I1 -sg[k][0]<0.00001):
                continueSG = False
            sg[k][1] = R0 / I0
            sg[k][0] = (I1-R1) / I1
            print('-------------------------------')
            print(str(k + 1) + '题目finish')

        print(str(kk ) +"次迭代，"+str(ni)+"个学生，20道题目的失误率和猜测率")
        print(sg)
        kk +=1
    endTime = time.time()
    print('********************************************')
    print('训练消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg



def testPredict():
    # 得到模型的参数，s和g
    sg = trainIDINAModel()
    startTime = time.time()
    print('预测开始时间：' + str(int(startTime)))
    # 测试集随机挑选20%的数据
    n = pd.read_csv('math2015/Math1/data.csv').sample(frac=0.2)
    # n = pd.read_csv('math2015/Math1/data.csv').head(headNum).sample(frac=0.2)
    Q = pd.read_csv('math2015/Math1/q.csv')
    # 536*0.2个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape
    # 预测的每个学生的技能向量
    a = np.zeros(ni,dtype=int)
    IL = np.ones((ni, 2 ** Qj))

    for l in range(2 ** Qj):
        # 学生的数量
        for i in range(ni):
            # 题目的数量
            for k in range(nj):
                # 计算Q矩阵中，第k题的情况下，l这个技能模式能否答对
                nl = nTrueOrFalse((Q.iloc[k]), l)
                # 理论上，答对的情况
                if nl == 1:
                    # 实际该学生，对于k这道题做对了，似然函数*(1-s)
                    if n.iloc[i][k] == 1:
                        IL[i][l] *= (1 - sg[k][0])
                    # 实际该学生，对于k这道题居然做错了，似然函数*s，失误率
                    else:
                        IL[i][l] *= sg[k][0]
                # 理论上，答错的情况
                else:
                    # 实际该学生，对于k这道题居然做对了，似然函数*g，猜测率
                    if n.iloc[i][k] == 1:
                        IL[i][l] *= sg[k][1]
                    # 实际该学生，对于k这道题确实做错了，似然函数*(1-g)
                    else:
                        IL[i][l] *= 1 - sg[k][1]


    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    for i in range(ni):
        v_max = 0
        v_max_index = 0
        for l in range(2 ** Qj):
            if IL[i][l] > v_max:
                v_max = IL[i][l]
                v_max_index = l
        a[i]=v_max_index
        #print(bin(v_max_index).replace('0b',''))
    print('测试集中各个学生的预测的a技能向量')
    print(a)


    # 计算准确率

    # 统计错误的数量
    countErr = 0
    for i in range(ni):
        for k in range(nj):
            # 预测的a值针对k这道题理论上的情况和实际如果不相同，累积+1
            if nTrueOrFalse(Q.iloc[k],a[i])!=n.iloc[i][k]:
                countErr +=1
    print('总共有'+str(ni)+'人，正确率为')
    print(1-countErr/(ni*nj))
    print('预测消耗时间：' + str(int(time.time())-int(startTime))+'秒')




'''
后续根据预测得到的a向量，来得到一个得分矩阵，判断准确率
以及提高更新迭代EM算法的sg部分的时间
'''

def main():
    testPredict()

if __name__ == "__main__":
    main()