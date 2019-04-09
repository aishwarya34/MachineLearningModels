import csv
import random
import math
import operator

# with training dataset , knn can use to make predictions
# and test dataset is used to evaluate the accuracy of predictions



def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename) as ifile:
        lines = csv.reader(ifile)
        dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(4):
            dataset[x][y] = float(dataset[x][y])    # first convert flower measures that were loaded as strings to numbers
        if random.random() < split:      # split data  in ratio  67:33
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])





def euclidean_distance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow(instance2[x]-instance1[x],2)
    return math.sqrt(distance)



def getNeighbour(trainingSet,testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(trainingSet[x],testInstance,length)
        distance.append((trainingSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distance[x])
    return neighbours


def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][0][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]



def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct +=1
    return  (correct/float(len(testSet))) * 100.0



# def main():


#prepare data
trainingSet = []
testSet = []
filename = 'iris.txt'
loadDataset(filename,0.67,trainingSet,testSet)
print("Train Data:  "+str(trainingSet))
print("Test data:  "+str(testSet))
#generate predictions
predictions = []
k = 3
for y in range(len(testSet)):
    neighbours = getNeighbour(trainingSet,testSet[y],k)
    result = getResponse(neighbours)
    predictions.append(result)
    print("predicted:  "+repr(result)+"    actual: "+repr(testSet[y][-1])   )

print("\n\naccuracy: "+repr(getAccuracy(testSet,predictions)) +"%")









