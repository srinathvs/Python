

import tensorflow as tf;
import tensorflow.keras;
import numpy as np;
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784', version=1)
mnist.keys();

X,y=mnist["data"], mnist["target"]
X.shape = (70000, 784)

y.shape = (70000,)


some_digit=X[0]
some_digit_image=some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
y = y.astype(np.uint8)
print(y[0]);


x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5= (y_train==5)
y_test_5= (y_test==5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

print(sgd_clf.predict([some_digit]))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds=StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]


    y_train_folds=y_train_5[train_index]
    x_test_fold=x_train[test_index]
    y_test_fold= y_train_5[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum( y_pred == y_test_fold)

    print(n_correct/len(y_pred))




"""
print(cv2.__version__);
img = cv2.imread("C:/Users/srina/Desktop/insurance card.jpg")  # Some random image to check if images are being read
edges=cv2.Canny(img,100,150)
cv2.imshow('sample', edges)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()
"""