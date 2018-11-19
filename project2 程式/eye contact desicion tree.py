from math import log
import operator





direction = 'D:\graduated_data\Datamining\project2\project_2_dataset_V2.txt'


###取得訓練資料,建立資料集與attribute

def createDataSet(direction):
    fileset = open(direction)
    #fileset.read()
    dataSet = [data.strip().split(' ') for data in fileset.readlines()]
    labels = dataSet[0]
    del(dataSet[0])
    return dataSet, labels

###計算entropy
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)						
    labelCounts = {}								
    for featVec in dataSet:						
        currentLabel = featVec[-1]					
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1				
    print("labelCounts =",labelCounts)
    shannonEnt = 0.0								
    for key in labelCounts:							
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)			
    return shannonEnt								

###劃分子資料集
def splitData(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#splitData(dataSet,0,1)
###選取最好的特徵(訊息增益) 有不同算法
###ID3訊息增益
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1					
    baseEntropy = calcShannonEnt(dataSet) 				
    bestInfoGain = 0.0  								
    bestFeature = -1									
    for i in range(numFeatures): 						
		#获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)     					
        newEntropy = 0.0  								
        for value in uniqueVals: 						
            subDataSet = splitData(dataSet, i, value) 		
            prob = len(subDataSet) / float(len(dataSet))   	
            #prob = float(len(subDataSet))/len(dataset)
            newEntropy += prob * calcShannonEnt(subDataSet) 
            #newEntropy  = prob * calcShannonEnt(subDataSet) 
            infoGain = baseEntropy - newEntropy 				
        print("第%d個特徵的增益為%.3f" % (i, infoGain))		
        if (infoGain > bestInfoGain): 						
            bestInfoGain = infoGain 							
            bestFeature = i 									
    return bestFeature 											

###統計classlist中出現最多的元素
def majorityCnt(classList):
    classCount = {}
    for vote in classList:	
        if vote not in classCount.keys():classCount[vote] = 0	
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)		
    return sortedClassCount[0][0]	
#majorityCnt(classList)

###建決策樹
def createTree(dataSet,labels, featLabels):    
    classList=[example[-1] for example in dataSet]  
    if classList.count(classList[0])==len(classList): 
        return classList[0]
    if len(dataSet[0])==1:  
        return majorityCnt(classList)
    #print("dataset=",dataSet)
    bestFeat=chooseBestFeatureToSplit(dataSet)  
    bestFeatLabel = labels[bestFeat]
    print("bestFeature=",bestFeatLabel)
    print("labels=",labels)
    featLabels.append(bestFeatLabel)
    feature=[example[bestFeat] for example in dataSet] 
    featValue=set(feature)  
    del(labels[bestFeat])
    Tree={bestFeatLabel:{}}  
    for value in featValue:
        subLabel=labels[:]
        Tree[bestFeatLabel][value]=createTree(splitData(dataSet,bestFeat,value),subLabel, featLabels) 
    return Tree


def classify(myTree, featLabels, testVec,labels2):
    firstStr = next(iter(myTree))		
    secondDict = myTree[firstStr]													
    #print("fs=",firstStr)
    ind=labels2.index(firstStr)    
    #print("index=",ind)
    secondDict = secondDict[testVec[ind]]													
    #print("sd=",secondDict)
    #featIndex = featLabels.index(firstStr)
    #print("fi=",featIndex)
      	
    if type(secondDict).__name__ == 'dict':
        #print("=dict")
        classLabel = classify(secondDict, featLabels, testVec,labels2)
    else: 
        #print("!=dict")
        classLabel = secondDict
        #print(classLabel)
    return classLabel

def testResult(result):
    if result == '1':
        print("1 : the patient should be fitted with hard contact lenses")
    if result == '2':
        print("2 : the patient should be fitted with soft contact lenses")
    if result == '3':
        print("3 : the patient should not be fitted with contact lenses")
        
###主程式
if __name__ == '__main__':
    dataSet, labels= createDataSet(direction)
    labels2=[]
    for i in labels:
        labels2.append(i)
    print("labels2=",labels2)
    #print(dataSet)
    #print(calcShannonEnt(dataSet))
    #print("最優特徵索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    testVec = ['1','1','1','2']
    result = classify(myTree, featLabels, testVec,labels2)
    testResult(result)
























