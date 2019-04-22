# Perceptron implementation
import util
import random

PRINT = True


class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    # def _predict(self, data, label):
    #     activation = 0.0
    #     for key, value in self.weights[label].items():
    #         activation += value * data[key]
    #
    #     return activation

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = trainingData[0].keys()  # could be useful later

        for key in self.weights:
            for feature in self.features:
                self.weights[key][feature] = random.uniform(0.0, 1.0)

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for itr in range(len(trainingData)):
                activations = dict()
                for key_1, value_1 in self.weights.items():
                    activation = 0.0
                    for key_2, value_2 in value_1.items():
                        activation += value_2 * trainingData[itr][key_2]
                    activations[key_1] = activation

                max_activation = 0

                for key, value in activations.items():
                    if max_activation <= value:
                        max_activation = value

                if max_activation == trainingLabels[itr]:
                    continue
                else:
                    for key, value in trainingData[itr].items():
                        self.weights[trainingLabels[itr]][key] = self.weights[trainingLabels[itr]][key] + value

                    other_classes = [key for key in self.weights]

                    other_classes.remove(trainingLabels[itr])

                    for cls in other_classes:
                        for key, value in trainingData[itr].items():
                            self.weights[cls][key] = self.weights[cls][key] - value

                # prediction = self._predict(trainingData[itr], trainingLabels[itr])
                # error = trainingLabels[itr] - prediction
                # for key, value in trainingData[itr].items():
                #     self.weights[trainingLabels[itr]][key] = self.weights[trainingLabels[itr]][key] + 0.1 * error * value

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in weights:
                         w_label1 - w_label2

        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
