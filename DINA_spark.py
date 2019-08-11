import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import col, pandas_udf,udf,PandasUDFType
from pyspark.sql.types import LongType,DoubleType

multi = False
threshold = 50000


def multiply_func(n, r):
    lll = ((1 - sg[0]) ** n * sg[0] ** (1 - n)) ** r * (sg[1] ** n * (
            1 - sg[1]) ** (1 - n)) ** (1 - r)
    IL[i][l] = lll.prod()
    return lll
#lll.prod()  连乘  
((1 - s) ** n * s ** (1 - n)) ** r * (g ** n * (1 - g) ** (1 - n)) ** (1 - r)
Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
multiply_pd = pandas_udf(multiply_func, returnType=DoubleType())
ndf = spark.read.load("math2015/FrcSub/data.csv",format="csv", sep=",", inferSchema="true", header="true")
ndf.select(multiply_pd(col("1"), col("2"))).show()
sg = pd.DataFrame(sg)


def EStep(IL,sg,n,r,k,i):
    base = 2**(k-2)  
    global l
    for l in range(i*base,(i+1)*base):  
        p_arr = np.array([])
        IL[:, l] = 
        p_arr = p_arr=n.foreach(f)
    return IL

def MStep(IL,n,r,k,i):
    base = 2**(k-2)
    n = n.toPandas().values
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
    print('start train...')
    ni= n.count()
    Qi, k = Q.shape
    nj = Qi
    global sg
    sg = np.zeros((nj, 2))
    global K
    K = np.mat(np.zeros((k, 2 ** k), dtype=int))
    for j in range(2 ** k):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[k - len(l) + i, j] = l[i]

    std = np.sum(Q, axis=1)
    global r
    r = (Q * K == std) * 1
  
    
    for i in range(nj):
        sg[i][0] = 0.2
        sg[i][1] = 0.2

    continueSG = True
    kk =1
    lastLX = 1
 
    while continueSG == True:
        IL = np.zeros((ni, 2 ** k))
        #rdd
        rddIL = spark.createDataFrame(pd.DataFrame(IL))
        if multi==True:         
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
       
        if abs(LX-lastLX)<threshold:
            continueSG = False

        lastLX = LX
        sg[:,1] = IR[1] / IR[0]
        sg[:,0] = (IR[2]-IR[3]) / IR[2]
        print(str(kk ) )
        print(" circles")
        kk +=1
    endTime = time.time()
    print('DINA train cost time ')
    print((int(endTime-startTime)))
    return sg,r

def predictDINA(n,Q,sg,r):
    startTime = time.time()
    print('predict start...')

    ni = n.count()
    Qi, Qj = Q.shape
    nj = Qi
    IL = np.zeros((ni, 2 ** Qj))
    rddIL = spark.createDataFrame(pd.DataFrame(IL))
    k = Qj

    if multi == True:
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, k, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item
    else:
        for l in range(2 ** Qj):
            lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
            IL[:, l] = lll.prod(axis=1)
    a = IL.argmax(axis=1)

    i, j = n.shape
    print('precision is :')
    p1 = np.sum((r[:, a] == n.T) * 1) / (i * j)
    print(p1)
    print('predict cost times')
    print(int(time.time()) - int(startTime))
    print('-----------------------------------------------')
    return p1


def testPredict(model,dataSet):
    if dataSet == 'FrcSub':
        # qdf = spark.read.load("math2015/FrcSub/q.csv",format="csv", sep=",", inferSchema="true", header="true")
        n = spark.read.load("math2015/FrcSub/data.csv",format="csv", sep=",", inferSchema="true", header="true")
        # Q = np.mat(qdf.toPandas())
        #n = pd.read_csv('math2015/FrcSub/data.csv').values
        global Q
        Q = np.mat(pd.read_csv('math2015/FrcSub/q.csv'))
    elif dataSet == 'Math1':
        n = pd.read_csv('math2015/Math1/data.csv').values
        Q = np.mat(pd.read_csv('math2015/Math1/q.csv').head(15).values)
    elif dataSet == 'Math2':
        n = pd.read_csv('math2015/Math2/data.csv').head(1000).values
        Q = np.mat(pd.read_csv('math2015/Math2/q.csv').head(16).values)

    (trainingData, testData) = n.randomSplit([0.7, 0.3])
    sg,r = trainDINAModel(trainingData,Q)
    precision = predictDINA(testData, Q, sg, r)
def main():
    startTime = time.time()
    testPredict('DINA','FrcSub')
    print('total time:')
    print(time.time()-startTime)
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DINA")\
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    main()
    
    #dataframe的列数
    # len(n.first().__fields__)
    
    
    # pddata = datadf.toPandas()
    
    