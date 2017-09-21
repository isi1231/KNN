# -*- coding: utf-8 -*-

#一个简单的测试用的文件，每完成一个部分就会在这个文件里测试一下，也可以直接在KNN.py中直接测试

import KNN
#import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#group,labels = KNN.createDataSet()

#print (KNN.classify0([0,0],group,labels,3))

#datingDataMat,datingLabels = KNN.file2matrix('datingTestSet2.txt')
#print (datingDataMat[0:5])
#print (datingLabels[0:20])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels)
#,15.0*np.array(datingLabels))
#plt.show()

#normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
#print (normMat)
#print (ranges)
#print (minVals)

KNN.classifyPerson()
