import json
import math
import sys
import time
import traceback
from multiprocessing import Pool

import MySQLdb
import numpy as np
import pandas as pd

# 打开数据库连接
db = MySQLdb.connect("localhost", "root", "yzd111", "lanmao", charset='utf8')
# 用来测试少量的数据，减少计算等待时间
multi = False
# sg迭代的阈值
threshold = 10

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
                # 学生的数量
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
    i,j = K[:, a].shape
    return np.array(K[:, a]).reshape(j,i)[0]

    # a2 = discrete(continuously(IL))

    # print('连续化向量')
    # print(continuous)
    # 计算准确率
    # i, j = n.shape
    # print('总共有' + str(ni) + '个人，a准确率为：')
    # p1 = np.sum((r[:, a] == n.T) * 1) / (i * j)
    # print(p1)
    # # print('总共有' + str(ni) + '个人，a2准确率为：')
    # # p1 = np.sum((r[:, a2] == n.T) * 1) / (i * j)
    # # print(p1)
    # print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    # print('-----------------------------------------------')
    # return p1

def getN(matchId):
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    sql = "SELECT result FROM nmatrix"
    d = []

    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        # 每个学生的答题记录
        for row in results:
            result = json.loads(row[0])
            if matchId in result.keys():
                list = result[matchId]
                # 里面的每道题目
                ll = []
                for l in list:
                    if (l['state'] == 'fail'):
                        ll.append(0)
                    else:
                        ll.append(1)
                d.append(ll)
    except:
        print("Error: unable to fecth data N")
    N = (pd.DataFrame(d).fillna(0)).values
    return N
def getQ(matchId):
    cursor = db.cursor()
    sql = "SELECT skill,matchId FROM qmatrix"
    Q = []
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            skill = json.loads(row[0])
            matchID = row[1]
            if matchId == str(matchID):
                ll = []
                for l in skill:
                    if (l['state'] == 'fail'):
                        ll.append(0)
                    else:
                        ll.append(1)
                Q.append(ll)
    except:
        traceback.print_exc()
        print("Error: unable to fecth data Q")
    Q = np.mat(pd.DataFrame(Q))
    return Q


def setParam(sg, matchId):
    cursor = db.cursor()
    sql = "SELECT problemId,matchId FROM qmatrix"
    problemList = {}
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        if row[1] not in problemList.keys():
            problemList[row[1]] = []
        problemList[row[1]].append(row[0])
    i = 0
    for problemId in problemList[int(matchId)]:
        # SQL 更新语句
        sql = "UPDATE qmatrix SET slip = %s,guess=%s WHERE matchId = %s and problemId=%s" % (sg[i][0],sg[i][1],matchId,problemId)
        i+=1
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            traceback.print_exc()
            # 发生错误时回滚
            db.rollback()


def getParam(matchId):
    cursor = db.cursor()
    sql = "SELECT slip,guess FROM qmatrix WHERE matchId = %s" % matchId
    cursor.execute(sql)
    results = cursor.fetchall()
    sg = np.array(results)
    return sg


def getUserN(userId, matchId):
    cursor = db.cursor()
    sql = "SELECT result FROM nmatrix WHERE userId = %s" % userId
    d = []
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        result = cursor.fetchone()
        # 每个学生的答题记录
        result = json.loads(result[0])
        if matchId in result.keys():
            list = result[matchId]
            # 里面的每道题目
            ll = []
            for l in list:
                if (l['state'] == 'fail'):
                    ll.append(0)
                else:
                    ll.append(1)
            d.append(ll)
    except:
        traceback.print_exc()
        print("Error: unable to fecth data userN")
    N = (pd.DataFrame(d).fillna(0)).values
    return N


def setUserSkill(userId, predictResult):
    cursor = db.cursor()
    sql = "SELECT name FROM skill "
    d = []
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        skills = cursor.fetchall()
        i=0
        for skill in skills:
            data = {}
            data['name'] =skill[0]
            k = 1 if predictResult[i] > 0 else -1
            data['state'] =str(50+k*5)
            i+=1
            d.append(data)
        data = json.dumps(d)
        print(data)
        sql = "UPDATE user SET skill = '%s' WHERE id = %s" % (data,str(userId))
        print(sql)
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # 发生错误时回滚
            traceback.print_exc()
            db.rollback()
    except:
        traceback.print_exc()
        print("Error: unable to fecth data setUserSkill")




def main():
    startTime = time.time()
    matchId = sys.argv[1]
    userId = sys.argv[2]
    N = getN(matchId)
    Q = getQ(matchId)

    sg, r ,K= trainDINAModel(N, Q)

    setParam(sg,matchId)
    sg = getParam(matchId)

    userN = getUserN(userId,matchId)
    predictResult = predictDINA(userN, Q, sg, r,K)
    setUserSkill(userId,predictResult)
    print('总时间:')
    print(time.time() - startTime)
    db.close()

if __name__ == "__main__":
    main()