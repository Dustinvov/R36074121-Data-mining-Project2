###導入必要的套件
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from math import log
import operator

###檔案路徑
direction = 'D:\graduated_data\Datamining\project2\mushroom-classification\mushrooms.csv'

###讀數檔案，創建資料集並將資料分成訓練集與測試集
def createDataSet(direction):
    #fileset = pd.read_csv(direction, sep = "\t")   #讀取檔案並以"tab"分割資料
    fileset = pd.read_csv(direction, sep = ",")     #讀取檔案並以"，"分割資料
    labelencoder = LabelEncoder()                   #將資料變為數值變量，以利於建模
    for col in fileset.columns:
        fileset[col] = labelencoder.fit_transform(fileset[col])
    attributes = fileset.columns.tolist()           #建立特徵(attribute)標籤
    dataSet = fileset.iloc[0:,0:].values.tolist()   #建立資料集
    answerSet = fileset['class'].tolist()           #建立答案集
    data_training, data_testing, answer_training, answer_testing = train_test_split(dataSet, answerSet, random_state=0, train_size=0.8)     #提取訓練集與測試集
    return attributes, dataSet, answerSet ,data_training, data_testing, answer_training, answer_testing


###計算Entropy
def calcEntropy(data_training):
    numEntires = len(data_training)                 #返回數據的行數			
    attributeCounts = {}								#保存每個特徵出現的次數
    for featVec in data_training:						#對每組特徵向量進行統計	
        currentattribute = featVec[-1]					
        if currentattribute not in attributeCounts.keys():	
            attributeCounts[currentattribute] = 0
        attributeCounts[currentattribute] += 1		#特徵計數		
    print("AttributeCounts =",attributeCounts)
    shannonEnt = 0.0                                #Entropy
    for key in attributeCounts:						#利用公式計算Entropy	
        prob = float(attributeCounts[key]) / numEntires	
        shannonEnt -= prob * log(prob, 2)	
    print("Entropy = ",shannonEnt)
    return shannonEnt


###按照給定特徵劃分數據(訓練)集
def splitData(data_training, axis, value):                              
    retDataSet = []                                                 #創建數據集列表
    for featVec in data_training:                                   #便利資料(訓練)集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                         #去除選取的特徵
            reducedFeatVec.extend(featVec[axis+1:])                 #將符合條件的資料添加回數據集
            retDataSet.append(reducedFeatVec)
    return retDataSet


###選擇最好的特徵
##計算信息增益(information gain)，選取information gain最大的特徵
def chooseBestFeatureToSplit(data_training):
    numFeatures = len(data_training[0]) - 1	                        #特徵數量
    print("FeaturesNum = ",numFeatures)				
    baseEntropy = calcEntropy(data_training)                        #計算數據的Entropy
    bestInfoGain = 0.0  								                #information gain
    bestFeature = -1									                   #最好的特徵索引值
    for i in range(numFeatures): 					                   #遍歷所有的特徵  
        featList = [example[i] for example in data_training]       #創建集合{},元素不能重複 
        uniqueVals = set(featList)     				               #創建集合{},元素不能重複
        newEntropy = 0.0  								               #計算條件下的Entropy   
        for value in uniqueVals:                                  #計算information gain  
            subDataSet = splitData(data_training, i, value) 		 #劃分子集
            prob = len(subDataSet) / float(len(data_training))    #計算子集的機率  
            newEntropy += prob * calcEntropy(subDataSet)          #利用公式計算條件下的Entropy  
            infoGain = baseEntropy - newEntropy 			       #information gain 
        print("第%d個特徵的增益為%.3f" % (i, infoGain))		
        if (infoGain > bestInfoGain): 						        #計算information gain
            bestInfoGain = infoGain 							
            bestFeature = i
    print("最優特徵值為:特徵",bestFeature) 								
    return bestFeature


###統計"classlist"中出現最多的元素(分類標籤 0:可食用, 1:不可食用)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                      #統計classlist中每個元素出現的次數
        if vote not in classCount.keys():classCount[vote] = 0	
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)	
    return sortedClassCount[0][0]


###創建決策數
def createTree(data_training,attributes, featattributes):    
    classList=[example[-1] for example in data_training]        #取分類標籤(0:可食用, 1:不可食用)
    if classList.count(classList[0])==len(classList): 
        return classList[0]
    if len(data_training[0])==1:                                #遍歷完所有特徵
        return majorityCnt(classList)
    #print("data_training=",data_training)
    bestFeat=chooseBestFeatureToSplit(data_training)            #選擇最好的特徵
    bestFeatattribute = attributes[bestFeat]                    #最好的特徵之標籤
    print("bestFeature=",bestFeatattribute)                     
    print("attributes=",attributes)
    featattributes.append(bestFeatattribute)
    feature=[example[bestFeat] for example in data_training]    #得到训练集中所有最好特徵(判斷節點)
    featValue=set(feature)                                      #去掉重複的值
    del(attributes[bestFeat])                                   #刪除已經使用過的特徵值
    Tree={bestFeatattribute:{}}                                 #根據最好的特徵標籤生成決策數
    for value in featValue:                                     #便利特徵值，生成決策樹
        subattribute=attributes[:]
        Tree[bestFeatattribute][value]=createTree(splitData(data_training ,bestFeat,value),subattribute, featattributes) #递归函数使得Tree不断创建分支，直到分类结束
    return Tree


###使用決策數進行分類
def classify(myTree, featattributes, testVec, attributeall):
    firstStr = next(iter(myTree))		
    secondDict = myTree[firstStr]													
    #print("fs=",firstStr)
    ind=attributeall.index(firstStr)    
    #print("index=",ind)
    secondDict = secondDict[testVec[ind]]													
    #print("sd=",secondDict)
    featIndex = featattributes.index(firstStr)
    #print("fi=",featIndex)
      	
    if type(secondDict).__name__ == 'dict':
        #print("=dict")
        classattribute = classify(secondDict, featattributes, testVec, attributeall)
    else: 
        #print("!=dict")
        classattribute = secondDict
        #print(classattribute)
    return classattribute


###測試準確率
def testingAccuracy(result, answer_testing):
    for i in range(0, len(testVec)-1):
        #print("i=",i)
        result.append(classify(myTree, featattributes, testVec[i], attributeall))
    accuracyNum = 0
    for i in range(0, len(testVec)-1):
        if result[i] == answer_testing[i]:
            accuracyNum = accuracyNum +1
    print("rightNum = " ,accuracyNum)
    print("totalNum = " ,len(result))
    accuracy = float(accuracyNum/len(result))
    print("accuracy = ",accuracy)
    return accuracy


###主程式
if __name__ == '__main__':
    attributes, dataSet, answerSet ,data_training, data_testing, answer_training, answer_testing = createDataSet(direction)
    attributeall=[]
    for i in attributes:
        attributeall.append(i)			
    featattributes = []
    myTree = createTree(data_training,attributes, featattributes)
    myTree
    print("DecisionTree = " ,myTree)
    testVec = data_testing                          #設定測試資料為資料(測試)集
    result = []
    testingAccuracy(result, answer_testing)      





















