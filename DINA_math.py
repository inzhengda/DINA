import numpy as np
import pandas as pd
import time
'''
使用公开数据集，math2015中的FrcSub
将data和q矩阵文件txt转为了csv的格式方便读取
'''





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
        return True
    else:
        return False



'''
计算每道题目的s失误率和g猜测率
'''
def calsg():
    startTime = time.time()
    print('开始时间：'+str(int(startTime)))

    #可以使用head函数控制读取的学生数量
    # n 表示每个学生每道题目的答题情况，1表示答对，0表示答错
    n = pd.read_csv('math2015/FrcSub/data.csv').head(10)
    # Q 表示每道题目答对需要掌握的知识点向量
    Q = pd.read_csv('math2015/FrcSub/q.csv')

    # 536个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape

    # 计算每道题目的s失误率和g猜测率
    sg = np.zeros((nj, 2))
    # 计算每个学生，在2^k中技能模式下面的似然函数，暂且将似然函数等于后验概率
    IL = np.ones((ni, 2**Qj))

    # 初始化每道题目的s失误率和g猜测率，
    # sg[i][0]表示第i道题目的s失误率，sg[i][1]表示第i道题目的g猜测率
    for i in range(nj):
        sg[i][0] = 0.1
        sg[i][1] = 0.1

    # 计算s和g迭代的次数，目前简化，只迭代1次
    for kk in range(1):
        # 技能模式的数量
        for l in range(2 ** Qj):
            # 学生的数量
            for i in range(ni):
                # 题目的数量
                for k in range(nj):
                    # 计算Q矩阵中，第k题的情况下，l这个技能模式能否答对
                    nl = nTrueOrFalse((Q.iloc[k]), l)
                    # 理论上，答对的情况
                    if nl == True:
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
        print("IL是所有学生，所有技能模式的似然概率矩阵")
        print(IL)

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
                    if nl ==True:
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
            sg[k][1] = R0 / I0
            sg[k][0] = (I1-R1) / I1
            print('-------------------------------')
            print(str(k + 1) + '题目finish')

        print(str(kk + 1) +"次迭代，"+str(ni)+"个学生，20道题目的失误率和猜测率")
        print(sg)
    endTime = time.time()
    print('********************************************')
    print('消耗时间：'+str(int(endTime-startTime))+'秒')
    print('********************************************')
    return sg,n,Q,IL

def cal():
    sg,n,Q,IL = calsg()
    # 536个学生,20道题目
    ni, nj = n.shape
    # 20道题目，8个知识点
    Qi, Qj = Q.shape
    # 预测的每个学生的技能向量
    a = np.zeros(ni)

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
    print('各个学生的预测a技能向量')
    print(a)
    return a

'''
后续根据预测得到的a向量，来得到一个得分矩阵，判断准确率
以及提高更新迭代EM算法的sg部分的时间

'''


def main():
    cal()

if __name__ == "__main__":
    main()