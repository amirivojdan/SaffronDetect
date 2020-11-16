import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


class SaffronDetect:

    def __init__(self, feature_configuration, max_iterations=1000):
        self.max_iteration = max_iterations
        self.samples = []
        self.labels = []
        self.train_samples = []
        self.train_labels = []
        self.test_samples = []
        self.test_labels = []
        self.feature_configuration = feature_configuration
        self.classifiers = []
        self.parameters = []

    def load_dataset(self, positive_samples_path, negative_samples_path):
        print("loading dataset")
        positive_samples = os.listdir(positive_samples_path)
        negative_samples = os.listdir(negative_samples_path)

        for sample in positive_samples:
            img = Image.open(positive_samples_path + '\\' + sample)
            gray = img.convert('L')
            feature_descriptor = hog(gray,
                                     self.feature_configuration.orientations,
                                     self.feature_configuration.cell_size,
                                     self.feature_configuration.normalization_block_size,
                                     block_norm='L2', feature_vector=True)
            self.samples.append(feature_descriptor)
            self.labels.append(1)

        for sample in negative_samples:
            img = Image.open(negative_samples_path + '\\' + sample)
            gray = img.convert('L')
            feature_descriptor = hog(gray,
                                     self.feature_configuration.orientations,
                                     self.feature_configuration.cell_size,
                                     self.feature_configuration.normalization_block_size,
                                     block_norm='L2', feature_vector=True)
            self.samples.append(feature_descriptor)
            self.labels.append(0)

        (trainData, testData, trainLabels, testLabels) = train_test_split(
            np.array(self.samples), self.labels, test_size=0.50, shuffle=True, random_state=0)

        self.train_samples = trainData
        self.train_labels = trainLabels

        self.test_samples = testData
        self.test_labels = testLabels

        print("we are done with dataset")

    def save_models(self):
        pass

    def add_classifier(self, classifier, parameters):
        self.classifiers.append(classifier)
        self.parameters.append(parameters)

    def load_models(self):
        pass

    def train_classifier(self, scoring='f1_macro'):

        (trainData, testData, trainLabels, testLabels) = train_test_split(
            np.array(self.samples), self.labels, test_size=0.50, shuffle=True, random_state=0)

        self.train_samples = trainData
        self.train_labels = trainLabels

        print("training just Started! ")
        i = 0
        for classifier in self.classifiers:
            clf = GridSearchCV(classifier, self.parameters[i], scoring=scoring)
            clf.fit(self.train_samples, self.train_labels)
            predictions = clf.predict(testData)
            print(classification_report(testLabels, predictions))
            self.classifiers[i] = clf.best_estimator_
            i = i+1
        print("Well Done Bro! ")

    def evaluate(self):
        pass

    def plot_roc_curves(self):
        ax = plt.gca()
        for classifier in self.classifiers:
            plot_roc_curve(classifier, self.test_samples, self.test_labels, ax=ax)
        plt.show()
        pass
