from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential, Model, load_model 
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import KFold

IMG_WIDTH, IMG_HEIGHT = 224,224
BATCH_SIZE = 32
EPOCHS = 5
N_CLASSES = 1
TRAIN_DATA_PATH = '/content/drive/MyDrive//DATASET2/'


generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input,
                                  
                                                              
                                        )


data=generator.flow_from_directory(TRAIN_DATA_PATH,
                                          target_size=(IMG_WIDTH, IMG_HEIGHT),
                                          batch_size=2000, 
                                          shuffle=False,
                                          class_mode="categorical",
                                          )


X, y = data.next()
print(X.shape)
print(y.shape)

restnet = tf.keras.applications.resnet50.ResNet50 (include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT,3))
output = restnet.layers[-1].output
output = Flatten()(output)
restnet = Model(restnet.input, output)

for layer in restnet.layers:
    layer.trainable = False

kfold = KFold(n_splits=3, shuffle=True)
fold_no = 0
acc_per_fold = []
loss_per_fold = []
historys = []
s = []
for train, test in kfold.split(X, y):
  model = Sequential()
  model.add(restnet)
  model.add(Dense(4096 , activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4096 , activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(2, activation='softmax'))
  model.add(Dense(N_CLASSES, activation='softmax'))
  model.compile(optimizer=Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['acc'])

  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(X[train], y[train],
              batch_size=32,
              epochs=EPOCHS,
              verbose=1)
  historys.append(history)

  # Generate generalization metrics
  scores = model.evaluate(X[test], y[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  s.append(scores)

  model.save(f"/content/drive/My Drive/model_{fold_no}.h5")
  with open(f'/content/drive/My Drive/history_{fold_no}', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


  # Increase fold number
  fold_no = fold_no + 1

print("acc_per_fold")
print(acc_per_fold)
print("loss_per_fold")
print(loss_per_fold)
print(s)


import matplotlib.pyplot as plt

for history in historys:
 plt.plot(history.history['acc'])
 plt.title('model accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train1','train2','train3'], loc='upper left')

 for history in historys:
   plt.plot(history.history['loss'])
   plt.title('model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['loss1', 'loss2','loss3'], loc='upper left')


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

testing_data_path = '/content/drive/My Drive/DATASET/newdata'
model_path = '/content/drive/My Drive/model_3.h5' #exemple

model = load_model(model_path)
img_width, img_height = 224, 224
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
testing_generator = test_datagen.flow_from_directory(
    testing_data_path,
    shuffle=False,
    target_size=(img_height, img_width),
    class_mode='categorical')

proba_pred = model.predict(testing_generator)
predicted = np.argmax(proba_pred, axis=1)
expected = testing_generator.classes
class_names = testing_generator.class_indices.keys()


def plot_confusion_matrix(cm, classes, title, normalized):
    col = plt.cm.GnBu
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.0f'
    else:
        fmt = '.2f'
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=col)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(expected, predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='confusion matrix', normalized=False)
plt.show()
