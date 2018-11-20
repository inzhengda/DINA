from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
import time
def EStep(IL,sg,n,r,k,i):
    print(os.getpid())
    base = 2**(k-1)
    for l in range(i*base,(i+1)*base):
        # 学生的数量
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL

def single(n,Q):
    s, k = Q.shape
    ni, nj = n.shape
    IL = np.zeros((ni, 2 ** k))
    K = np.mat(np.zeros((k, 2 ** k), dtype=int))
    for j in range(2 ** k):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[k - len(l) + i, j] = l[i]
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1
    sg = np.zeros((s, 2))
    for i in range(s):
        sg[i][0] = 0.01
        sg[i][1] = 0.01
    for l in range(2**k):
        # 学生的数量
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL

def multi(n,Q,nThreads):
    with Pool(processes=nThreads) as pool:
        s,k = Q.shape
        ni,nj = n.shape
        IL = np.zeros((ni, 2 ** k))
        K = np.mat(np.zeros((k, 2 ** k), dtype=int))
        for j in range(2 ** k):
            l = list(bin(j).replace('0b', ''))
            for i in range(len(l)):
                K[k - len(l) + i, j] = l[i]
        std = np.sum(Q, axis=1)
        r = (Q * K == std) * 1
        sg = np.zeros((s, 2))
        for i in range(s):
            sg[i][0] = 0.01
            sg[i][1] = 0.01
        multiple_results = [pool.apply_async(EStep, (IL,sg,n,r,k,i)) for i in range(nThreads)]
        for item in ([res.get(timeout=10000) for res in multiple_results]):
            IL += item
        return IL

if __name__ == '__main__':
    # start 4 worker processes
    n = pd.read_csv('math2015/Math2/data.csv').sample(frac=0.8).values
    Q = np.mat(pd.read_csv('math2015/Math2/q.csv').head(16).values)
    startTime = time.time()
    nThreads = 2
    IL1 = multi(n,Q,nThreads)
    # IL2 = single(n,Q)
    endTime = time.time()
    print(endTime-startTime)