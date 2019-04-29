from classificationMethod import *
import math
import operator
import util


class KNN(ClassificationMethod):

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.trainingData = None
        self.trainingLabels = None



    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels

    def classify(self, data):
        guesses = []
        for datum in data:
            guesses.append(self._getNeighbors(self.trainingData, datum, 15))
        return guesses

    def _euclideanDistance(self, instance1, instance2):
        distance = 0
        for key, value in instance1.items():
            distance += pow((value - instance2[key]), 2)
        return math.sqrt(distance)

    def _getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        for index in range(len(trainingSet)):
            dist = self._euclideanDistance(testInstance, trainingSet[index])
            distances.append((self.trainingLabels[index], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        classVotes = {}
        print neighbors
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)

        return sortedVotes[0][0]
