# -*- coding: utf-8 -*-
import numpy as np
import operator
import os
#用来创建测试用的数据，可以删除
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#KNN分类算法实现
def classify0(inX,dataSet,labels,k):
    dataSetsize = dataSet.shape[0]  #shape[0]获取矩阵的行数，shape[1]获取矩阵的列数
    diffMat = np.tile(inX,(dataSetsize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances =sqDiffMat.sum(axis = 1)   #axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#将list中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

 #读取文件内容   
def file2matrix(filename):
    fr = open(filename)
    array0Lines=fr.readlines() #读取全部文件内容，每一行是一个元素
    numberoflines = len(array0Lines) #计算多少行
    returnMat = np.zeros((numberoflines,3)) #存储特征
    classLabelVector=[]  #存储类型
    index = 0
    for line in array0Lines:
        line = line.strip()   #去掉行首行尾删除空白符（包括'\n', '\r',  '\t',  ' ')
        listFromLine = line.split('\t')   #将整行数据分割成一个元素列表
        returnMat[index,0:3]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector

#归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    #print (minVals)
    maxVals = dataSet.max(0) 
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,1)
    return normDataSet,ranges,minVals

#测试函数
def DataClassTest():
    Ratio = 0.05;
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = datingDataMat.shape[0]
    numTestVecs = int(m*Ratio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with:%d,the real label is :%d"%(classifierResult,datingLabels[i]))
        if (classifierResult !=datingLabels[i]):
            errorCount +=1.0;
        
    print ("the total error rate is :%f" %(errorCount/float(numTestVecs)))
    
#交互式测试函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large dases']
    percentTats = float(input("percentage of time spent playing video games?:"))
    ffMiles = float(input("frequent flier miles earned per year?:"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("you wile probably like this person:",resultList[classifierResult - 1])
    
      
    
#classifyPerson()  
    
####################################################
# 测试0-1图像识别实现代码
def img2vector(filename):
    returnVec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j]=int(lineStr[j])        
    return returnVec
    
#testVector = img2vector('testDigits/0_13.txt')
#print (testVector[0,0:31])
    
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        filename = trainingFileList[i]
        fileStr = filename.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = img2vector('trainingDigits\%s'%(filename))
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenames = testFileList[i]
        fileStr = filenames.split('.')[0]
        TestNum = int(fileStr.split('_')[0])
        testVector = img2vector('testDigits\%s'%(filenames))
        classiferResult = classify0(testVector,trainingMat,hwLabels,3)
        print ("the true num is %d,the classifer num is %d"%(TestNum,classiferResult))
        if(classiferResult !=TestNum):
            errorCount +=1.0
    print ("\n the total number of errors is :%d"%(errorCount))
    print ("\nthe total error rate is:%f"%(errorCount/float(mTest)))
    
    
    
handwritingClassTest()
    
    
    
    
    
    
    
    