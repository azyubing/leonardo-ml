import math
import csv
import random
import operator


def distance(i1,i2):
    distance = 0
    for x in range(4):
        distance += pow((i1[x]-i2[x]),2)
    return math.sqrt(distance)


def getNeighbors(trainingSet,testInstance,k):
    distances=[]
    for i in range(len(trainingSet)):
        dis = distance(testInstance,trainingSet[i])
        distances.append((trainingSet[i],dis))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def getResult(neighbors):
    votes = {}
    for i in range(len(neighbors)):
        result = neighbors[i][-1]
        # print("result is:", result)
        if result in votes:
            votes[result] += 1
        else:
            votes[result] = 1
    print(votes)
    sortedVotes = sorted(votes, key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0]



def main():
    trainingSet=[]
    testSet=[]
    splitRatio=0.75
    filename=r"/Users/sap/Desktop/Python/algorithm/irisdata.txt"
    with open(filename,'rt') as datafile:
        lines = csv.reader(datafile)
        dataSet=list(lines)
        print("dataset is: ", dataSet)
        for x in range(len(dataSet)-1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            if(random.random() < splitRatio):
                trainingSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])

    print("trainingSet len:", len(trainingSet))
    print("testSet len:", len(testSet))






    results=[]
    for i in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[i],3)
        result = getResult(neighbors)
        results.append(result)
        print("期望值:",testSet[i][-1], "  实际值:",result)

    correct = 0
    for i in range(len(results)):
        if(results[i] == testSet[i][-1]):
            correct += 1
        print("准确率: ",correct/float(len(results))*100, "%")




main()