import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
        print("Loading dataset...")
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
        print("We are done with the dataset.")

    def save_models(self):
        pass

    def add_classifier(self, classifier, parameters):
        self.classifiers.append(classifier)
        self.parameters.append(parameters)

    def load_models(self):
        pass

    def pre_processing(self, test_size=0.2):

        (train_samples, test_samples, train_labels, test_labels) = train_test_split(
            np.array(self.samples), self.labels, test_size=test_size, shuffle=True, random_state=0)

        self.train_samples = train_samples
        self.train_labels = train_labels

        self.test_samples = test_samples
        self.test_labels = test_labels

        standard_scaler = StandardScaler()
        self.train_samples = standard_scaler.fit_transform(self.train_samples)
        self.test_samples = standard_scaler.transform(self.test_samples)

        # print(self.train_samples.shape)
        pca = PCA(n_components=0.99, whiten=True)
        self.train_samples = pca.fit_transform(self.train_samples)
        # print(self.train_samples.shape)
        self.test_samples = pca.transform(self.test_samples)

    def train_classifier(self, scoring='f1_macro'):

        print("Training just Started! ")
        i = 0
        for classifier in self.classifiers:
            clf = GridSearchCV(classifier, self.parameters[i], scoring=scoring)
            clf.fit(self.train_samples, self.train_labels)
            predictions = clf.predict(self.test_samples)
            print(classification_report(self.test_labels, predictions))
            self.classifiers[i] = clf.best_estimator_
            i = i+1

        print("Training Finished!")

    def evaluate(self):
        pass

    def plot_roc_curves(self):
        ax = plt.gca()
        for classifier in self.classifiers:
            plot_roc_curve(classifier, self.test_samples, self.test_labels, ax=ax)
        plt.show()
