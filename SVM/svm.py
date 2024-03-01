from google.colab import drive
drive.mount('/content/drive')

# import the required modules
import os
from glob import glob

import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import cv2


def show_img(path):

    img = cv2.imread(path)
    #cv2.imshow("sdsdsd",img)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hist_b = np.histogram(b,bins=100)[0]
    hist_g = np.histogram(g,bins=100)[0]
    hist_r = np.histogram(r,bins=100)[0]
    #plt.plot(hist_r, color='r', label="r")
    #plt.plot(hist_g, color='g', label="g")
    #plt.plot(hist_b, color='b', label="b")
    #plt.legend()
    #plt.show()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    hist_h = np.histogram(h,bins=100)[0]
    hist_s = np.histogram(s,bins=100)[0]
    hist_v = np.histogram(v,bins=100)[0]
    #plt.plot(hist_h, color='r', label="h")
    #plt.plot(hist_s, color='g', label="s")
    #plt.plot(hist_v, color='b', label="v")
    #plt.legend()
    #cv2.waitKey(3000)

    #plt.show()

    return hist_r, hist_g, hist_b, hist_h, hist_s, hist_v
# set the path to your dataset
path = '/content/drive/My Drive/SVMDATASET/'

# set the base for entropy calculation
base = 3

# initialize the lists to store the descriptors and labels
descriptors = []
labels = []

# read the melanoma images and extract their descriptors and labels
melanoma_images = sorted(glob(os.path.join(path, "MELANOMESEG", "*")))
for image_path in melanoma_images:
    # calculate the descriptor
    r, g, b, h, s, v = show_img(image_path)
    descriptor = []
    pk = np.array(r)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(g)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(b)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(h)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(s)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(v)
    H = entropy(pk, base=base)
    descriptor.append(H)
    descriptors.append(descriptor)
    labels.append(1)  # melanoma label is 1

# read the naevus images and extract their descriptors and labels
naevus_images = sorted(glob(os.path.join(path, "NAEVUSSEG", "*")))
for image_path in naevus_images:
    # calculate the descriptor
    r, g, b, h, s, v = show_img(image_path)
    descriptor = []
    pk = np.array(r)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(g)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(b)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(h)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(s)
    H = entropy(pk, base=base)
    descriptor.append(H)
    pk = np.array(v)
    H = entropy(pk, base=base)
    descriptor.append(H)
    descriptors.append(descriptor)
    labels.append(0)  # naevus label is 0

# convert the lists to numpy arrays
X = np.array(descriptors)
y = np.array(labels)

# split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import csv

# create a list to hold the data
data = []

# iterate over the images and their descriptors
for i, descriptor in enumerate(descriptors):
    label = labels[i]
    row = [label] + descriptor
    data.append(row)

# write the data to a CSV file
with open('/content/drive/My Drive/descriptors1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['label', 'hist_r', 'hist_g', 'hist_b', 'hist_h', 'hist_s', 'hist_v'])
    for row in data:
        writer.writerow(row)

# calculate class weights
total_samples = len(y_train)
num_melanoma_samples = np.sum(y_train == 1)
num_nevus_samples = total_samples - num_melanoma_samples
class_weight = {0: 1, 1: num_nevus_samples / num_melanoma_samples * 5}

# create an SVM model with class weights
svm_model = SVC(kernel='linear', class_weight=class_weight)

# train the SVM model
history = svm_model.fit(X_train, y_train)

# make predictions on the test set
y_pred = svm_model.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
print('Confusion matrix:\n', cm)
print("Accuracy:", accuracy)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# predict probabilities for positive class
y_scores = history.decision_function(X_test)

# calculate fpr, tpr and thresholds for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# calculate AUC
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle

class HistoryCallback():
    def __init__(self):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def __call__(self, clf, X_train, y_train, X_val, y_val):
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        loss_train = np.mean(y_pred_train != y_train)
        loss_val = np.mean(y_pred_val != y_val)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        self.history['loss'].append(loss_train)
        self.history['accuracy'].append(accuracy_train)
        self.history['val_loss'].append(loss_val)
        self.history['val_accuracy'].append(accuracy_val)

# Load iris dataset and split into train/test
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, stratify=iris.target)

# Create SVM classifier
clf = SVC(kernel='linear', C=1)

# Create callback and fit SVM classifier with callback
callback = HistoryCallback()
clf.fit(X_train, y_train, callback=lambda x: callback(x, X_train, y_train, X_test, y_test))

# Save the history with pickle
with open('svm_historyffff.pkl', 'wb') as f:
    pickle.dump(callback.history, f)

# Plot the history
import matplotlib.pyplot as plt

plt.plot(callback.history['loss'], label='train_loss')
plt.plot(callback.history['val_loss'], label='val_loss')
plt.legend()
plt.title('SVM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(callback.history['accuracy'], label='train_acc')
plt.plot(callback.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('SVM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


