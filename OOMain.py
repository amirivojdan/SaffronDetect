import numpy as np
import os

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from FeatureConfiguration import FeatureConfiguration
from SaffronDetect import SaffronDetect

print("Hi")

feature_configuration = FeatureConfiguration()

saffron_detector = SaffronDetect(feature_configuration)
saffron_detector.load_dataset(r"PositiveSamples", r"NegativeSamples")
saffron_detector.pre_processing(test_size=0.2)

random_forest_model = RandomForestClassifier(n_estimators=3)
KNN_model = KNeighborsClassifier(n_neighbors=100, metric='chebyshev')
svm_model = SVC(kernel='poly', random_state=0, C=100, gamma=5)

# Set the parameters by cross-validation
knn_parameters = {'n_neighbors': [7, 10, 20],
                  'metric': ['euclidean', 'chebyshev', 'manhattan']}
saffron_detector.add_classifier(KNN_model, knn_parameters)

svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
saffron_detector.add_classifier(svm_model, svm_parameters)

random_forest_parameters = {'n_estimators': [3, 7, 10, 20, 50, 100, 300]}
saffron_detector.add_classifier(random_forest_model, random_forest_parameters)

saffron_detector.train_classifier()
saffron_detector.plot_roc_curves()
