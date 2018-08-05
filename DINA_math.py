import numpy as np
import os
import pandas as pd
from collections import Counter

'''
使用公开数据集，math2015
'''


'''
判别q这道题目对应的l技能模式下能否做对题目
将q向量转为2进制的字符串，再转为int整数
'''
def nTrueOrFalse(q1, l):
    q =int(''.join(map(str, q1)), 2)
    if q&l == q:
        return True
    else:
        return False

'''
计算每道题目的s失误率和g猜测率
'''
import  time
def calsg():
    startTime = time.time()
    print('开始时间：'+str(int(startTime)))
    n = pd.read_csv('math2015/FrcSub/data.csv').head(100)
    Q = pd.read_csv('math2015/FrcSub/q.csv')

    # 536个学生,20道题目
    n1, m1 = n.shape
    # 20道题目，8个知识点
    n2, m2 = Q.shape

    sg = np.zeros((m1, 2))
    IL = np.ones((n1, 2**m2))
    for i in range(m1):
        # for j in range(2):
        sg[i][0] = 0.1
        sg[i][1] = 0.1



    # 每道题目迭代的次数
    for kk in range(1):
        #题目的数量

        for l in range(2 ** m2):
            # 总共I个学生
            for i in range(n1):
                for k in range(m1):
                    nl = nTrueOrFalse((Q.iloc[k]), l)
                    if nl == True:
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= (1 - sg[k][0])
                        else:
                            IL[i][l] *= sg[k][0]
                    else:
                        if n.iloc[i][k] == 1:
                            IL[i][l] *= sg[k][1]
                        else:
                            IL[i][l] *= 1 - sg[k][1]
        print("20道题目的得分概率矩阵")
        print(IL)


        for k in range(m1):
            I0 = 0
            R0 = 0
            I1 = 0
            R1 = 0
            #总共L种技能模式
            for l in range(2 ** m2):
                #总共I个学生
                for i in range(n1):
                    nl = nTrueOrFalse((Q.iloc[k]),l)
                    if nl ==True:
                        I1 += IL[i][l]
                        if n.iloc[i][k] ==1:
                            R1 += IL[i][l]
                    else:
                        I0 += IL[i][l]
                        if n.iloc[i][k] ==1:
                            R0 += IL[i][l]
            sg[k][1] = R0 / I0
            sg[k][0] = (I1-R1) / I1
            print('-------------------------------')
            print(str(k + 1) + '题目finish')

        print(str(kk + 1) +"次迭代，"+str(n1)+"个学生，20道题目的失误率和猜测率")
        print(sg)
    endTime = time.time()
    print('********************************************')
    print('消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg,n,Q,IL

def cal():
    sg,n,Q,IL = calsg()
    # 536个学生,20道题目
    n1, m1 = n.shape
    # 20道题目，8个知识点
    n2, m2 = Q.shape
    a = np.zeros(n1)
    for i in range(n1):
        v_max = 0
        v_max_index = 0
        for l in range(2 ** m2):
            if IL[i][l] > v_max:
                v_max = IL[i][l]
                v_max_index = l
        a[i]=v_max_index
        #print(bin(v_max_index).replace('0b',''))
    print('各个学生的预测a技能向量')
    print(a)
    return a

cal()