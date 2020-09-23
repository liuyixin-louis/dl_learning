#coding:utf-8

from numpy import *
import operator

class KNN:
    def __init__(self):
        super().__init__()
    
    ##给出训练数据以及对应的类别
    def createDataSet(self):
        group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
        labels = ['A','A','B','B']
        return group,labels

    ###通过KNN进行分类
    def classify(self,input,dataSet,label,k):
        dataSize = dataSet.shape[0]
        ####计算欧式距离
        diff = tile(input,(dataSize,1)) - dataSet
        sqdiff = diff ** 2
        squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
        dist = squareDist ** 0.5
        
        ##对距离进行排序
        sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

        classCount={}
        for i in range(k):
            voteLabel = label[sortedDistIndex[i]]
            ###对选取的K个样本所属的类别个数进行统计
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        ###选取出现的类别次数最多的类别
        maxCount = 0
        for key,value in classCount.items():
            if value > maxCount:
                maxCount = value
                classes = key

        return classes    


if __name__ == '__main__':
    knn = KNN()
    dataSet,labels = knn.createDataSet()
    input = array([1.1,0.3])
    K = 3
    output = knn.classify(input,dataSet,labels,K)
    print("测试数据为:",input,"分类结果为：",output)