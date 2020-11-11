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

orientations = 9
cellSize = (8, 8)
normalizationBlockSize = (2, 2)
threshold = 0.3

positiveSamples = os.listdir(r"PositiveSamples")
negativeSamples = os.listdir(r"NegativeSamples")

data = []
labels = []

for sample in positiveSamples:
    img = Image.open(r"PositiveSamples" + '\\' + sample)
    gray = img.convert('L')
    featureDescriptor = hog(gray, orientations, cellSize, normalizationBlockSize, block_norm='L2', feature_vector=True)
    data.append(featureDescriptor)
    labels.append(1)

for sample in negativeSamples:
    img = Image.open(r"NegativeSamples" + '\\' + sample)
    gray = img.convert('L')
    featureDescriptor = hog(gray, orientations, cellSize, normalizationBlockSize, block_norm='L2', feature_vector=True)
    data.append(featureDescriptor)
    labels.append(0)

(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), labels, test_size=0.20, shuffle=True, random_state=0)

linearSVMModel = LinearSVC(max_iter=3000)
randomForestModel = RandomForestClassifier(n_estimators=3)
KNNModel = KNeighborsClassifier(n_neighbors=100, metric='chebyshev')
svm = SVC(kernel='poly', random_state=0, C=100, gamma=5)

linearSVMModel.fit(trainData, trainLabels)
randomForestModel.fit(trainData, trainLabels)
KNNModel.fit(trainData, trainLabels)
svm.fit(trainData, trainLabels)

LinearSVMPredictions = linearSVMModel.predict(testData)
randomForestPredictions = randomForestModel.predict(testData)
KNNPredictions = KNNModel.predict(testData)
svmPredictions = svm.predict(testData)

print(classification_report(testLabels, KNNPredictions))

TN, FP, FN, TP = confusion_matrix(testLabels, KNNPredictions).ravel()

# TPR = TP / (TP + FN)  # Sensitivity,recall
# TNR = TN / (TN + FP)  # Specificity
# PPV = TP / (TP + FP)  # Precision
# ACC = (TP + TN) / (TP + FP + FN + TN)  # Accuracy

# print('TN:' + str(TN) + ' FP:' + str(FP) + ' FN:' + str(FN) + ' TP:' + str(TP))
# print('Sensitivity:' + str(TPR))
# print('Specificity:' + str(TNR))
# print('Precision:' + str(PPV))
# print('Accuracy:' + str(ACC))

classifiers = [linearSVMModel, randomForestModel, KNNModel, svm]
ax = plt.gca()
for i in classifiers:
    plot_roc_curve(i, testData, testLabels, ax=ax)
plt.show()
