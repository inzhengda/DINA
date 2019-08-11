import numpy as np
import os
import pandas as pd
from collections import Counter

'''
源代码部分，筛选出成功执行完的id
Java 大项目地数据
'''
root_path = 'D:\demo\DINA\judgelog'


def select_success():
    success_list = []
    for id in os.listdir(root_path):
        if 'source' in os.listdir(root_path + '\\' + id):
            success_list.append(id)
    print('success total :' + str(len(success_list)))
    f = open('success_id.txt', 'w')
    f.write(','.join(success_list))
    return success_list
select_success()
def loadDataSet():
    dataSet = []
    fr = open('m_judge.csv', encoding='utf-8')
    f = open('success_id.txt', 'r')
    success_list = f.readline()
    success_list = success_list.split(',')
    for line in fr.readlines():
        curLine = line.split(',')
        id = curLine[0].replace('"', '')
        if id not in success_list:
            continue
        groupId = int(curLine[-1].replace('"', ''))
        costTime = int(curLine[-2].replace('"', ''))
        pushTime = int(curLine[4].split(':')[-1])
        pullTime = int(curLine[5].split(':')[-1])
        fltLine = [int(id),groupId, pushTime, pullTime, costTime]
        dataSet.append(fltLine)
    data  = pd.DataFrame(dataSet,columns=('id','groupId','push','pull','cost'))
    data2 = data.sort_values("cost")
    # print(data2["id"].head(100))
    data2.to_csv("data_cost.csv")
loadDataSet()
'''
根据时间划分3个阶段，每个阶段当作一道题目
筛选出包的集合当作技能集合
得到Q矩阵
'''
def Qmatrix():
    d = {}
    total_A = list()
    total_B = list()
    total_C = list()
    data = pd.read_csv("data_cost.csv")
    success_list = data["id"].values
    costTime = data["cost"].values
    for id,cost in zip(success_list,costTime):
        s = set()
        files = os.listdir(root_path + '\\' + str(id) + '\source\src\main\java\pku')
        for file in files:
            if file.endswith('.java'):
                fr = open(root_path + '\\' + str(id) + '\source\src\main\java\pku\\' + file, encoding='utf-8')
                for line in fr.readlines():
                    if 'public' in line:
                        break
                    if 'import' not in line:
                        continue
                    package_name = line.split(' ')[1]
                    if package_name.startswith('java'):
                        s.add(package_name.replace(";", "").replace("\n", ""))
                        if cost<150000:
                            total_A.append(package_name.replace(";", "").replace("\n", ""))
                        elif cost <300000:
                            total_B.append(package_name.replace(";", "").replace("\n", ""))
                        else :
                            total_C.append(package_name.replace(";", "").replace("\n", ""))
        d[id] = list(s)

    # print(Counter(total_A).most_common(20))
    # print(Counter(total_B).most_common(20))
    # print(Counter(total_C).most_common(20))
    lA = list()
    for t_A in Counter(total_A).most_common(20):
        lA.append(t_A[0])
    lB = list()
    for t_B in Counter(total_B).most_common(20):
        lB.append(t_B[0])
    lC = list()
    for t_C in Counter(total_C).most_common(20):
        lC.append(t_C[0])
    # print(lA)
    # print(lB)
    # print(lC)
    allABC = set()
    justA =set()
    justB = set()
    justC = set()
    justAB = set()
    justBC = set()
    for item in lA:
        if item not in lB :
            if item not in lC:
                justA.add(item)
        else:
            if item in lC:
                allABC.add(item)
            else :
                justAB.add(item)
    for item in lB:
        if item not in lA:
            if item not in lC:
                justB.add(item)
            else:
                justBC.add(item)
    for item in lC:
        if item not in lA:
            if item not in lB:
                justC.add(item)
    # print(allABC)
    # print(justA)
    # print(justAB)
    # print(justB)
    # print(justBC)
    # print(justC)
    # k_set= allABC|justA|justAB|justB|justBC|justC
    # A_set = allABC|justA|justAB
    # B_set = allABC|justB|justBC
    # C_set = allABC|justC
    k_set =  justA |  justB |  justC
    A_set =  justA
    B_set =  justB
    C_set =  justC
    l = list(k_set)
    l.sort()
    Q = np.zeros((3,len(l)),dtype=int)
    for i in range(3):
        for j in range(len(l)):
            if i == 0:
                if l[j] in A_set:
                    Q[i][j] =1
            elif i==1:
                if l[j] in B_set:
                    Q[i][j] =1
            else :
                if l[j] in C_set:
                    Q[i][j] = 1

    print(l)
    print('Q矩阵')
    print(Q)
    # 计算每个学生出现的技能
    # vec = np.zeros((len(success_list), len(l)))
    # for i in range(len(success_list)):
    #     id = success_list[i]
    #     for j in range(len(l)):
    #         if l[j] in d[id]:
    #             vec[i][j] = 1
    # print(vec)

    # l是技能list集合，Q是题目矩阵
    return Q,l

'''
学生作答情况
'''

def Nmatrix():
    data = pd.read_csv('data_cost.csv')
    id = data['id'].values
    groupId1 = data['groupId'].values
    costTime = data['cost'].values
    d = dict()

    for id,groupId in zip(id,groupId1):
        if groupId not in d:
            d[groupId] = [id]
        else:
            d[groupId].append(id)

    # print([(k,d[k]) for k in sorted(d.keys())] )
    print(len(d))
    n = np.zeros((len(d.keys()), 3),dtype=int)
    l_group = list(sorted(d.keys()))

    for cost,groupId in zip(costTime,groupId1):
        k = l_group.index(groupId)
        if cost <115000:
            n[k][0] = 1
        elif cost < 275000:
            n[k][1] = 1
        else:
            n[k][2] = 1
    print("n矩阵")
    print(n)
    return n

'''
判别该道题目对应的l技能模式能否做对题目
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

def calsg():
    n = Nmatrix()
    # 63,3
    n1, m1 = n.shape
    Q, l = Qmatrix()
    # 3,9
    n2, m2 = Q.shape

    sg = np.zeros((3, 2))
    IL = np.zeros((3,n1, 2**m2))
    for i in range(3):
        for j in range(2):
            sg[i][j] = 0.1
    I0 = 0
    R0 = 0
    I1 = 0
    R1 = 0
    for kk in range(1):
        for k in range(3):
            for i in range(n1):
                for l in range(2**m2):
                    nl = nTrueOrFalse(list(Q[k]),l)
                    if nl ==True:
                        IL[k][i][l] = 1-sg[k][0]
                        I1 += IL[k][i][l]
                        if n[i][k] ==1:
                            R1 += IL[k][i][l]
                    else:
                        IL[k][i][l] = sg[k][1]
                        I0 += IL[k][i][l]
                        if n[i][k] ==1:
                            R0 += IL[k][i][l]
            sg[k][1] = R0 / I0
            sg[k][0] = (I1-R1) / I1
    print("三道题目的得分矩阵为1的概率")
    print(IL)
    print("三道题目的失误率和猜测率，只迭代了一次")
    print(sg)
    return sg

def cal():
    n = Nmatrix()
    # 63,3
    n1 ,m1= n.shape
    Q,l = Qmatrix()
    # 3,9
    n2,m2 = Q.shape

    a = np.ones((n1,2 **m2))
    sg = calsg()
    IL = np.zeros((3, n1, 2 ** m2))
    for i in range(n1):
        v_max = 0
        v_max_index = 0
        for l in range(2 ** m2):
            for k in range(3):
                nl = nTrueOrFalse(list(Q[k]), l)
                if nl == True:
                    IL[k][i][l] = 1 - sg[k][0]
                else:
                    IL[k][i][l] = sg[k][1]
                a[i][l] *=IL[k][i][l]
            if a[i][l] > v_max:
                v_max = a[i][l]
                v_max_index = l
        print(i)
        print(bin(v_max_index).replace('0b',''))
cal()
