from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
#,NASNetLarge
from keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.optimizers import Adam ,RMSprop


image_width, image_height = 224,224
TRAIN_DATA_PATH = 'dataset/Train'
TEST_DATA_PATH = 'dataset/Test'



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      shear_range=0.2,
      zoom_range=0.2,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
                    TRAIN_DATA_PATH,
                    batch_size=128,
                    class_mode='categorical',
                    shuffle=True,
                    target_size=(image_width, image_height)
)

test_generator =  test_datagen.flow_from_directory(
                    TEST_DATA_PATH,
                    batch_size=128,
                    class_mode='categorical',
                    shuffle=True,
                    target_size=(image_width, image_height)
)


train_steps = train_generator.n // train_generator.batch_size

print(train_steps)

test_steps = test_generator.n // test_generator.batch_size

print(test_steps)

rs = ResNet50(include_top=True,
                 weights= None,
                 input_tensor=None,
                 input_shape=(image_width, image_height, 3),
                 pooling='avg',
                 classes=2)


for layers in (rs.layers):
    layers.trainable = False

model = Sequential()

# Add the ResNet50 convolutional base model
model.add(rs)

# Add new layers
model.add(Flatten())
model.add(Dense(4096 , activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4096 , activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

lr = 1e-6
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
mcp = ModelCheckpoint('modelResNefinal.h5', verbose=1)

es = EarlyStopping(patience=2,verbose=1)
history = model.fit(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=10,validation_data=test_generator,
                    validation_steps=test_steps,
                    verbose=1,
                    callbacks=[mcp,es])
model.save("renet50finalmodele.h5")
model.evaluate(test_generator, verbose=1, steps=test_steps)


import matplotlib.pyplot as plt

accuracy      = history.history['accuracy']
val_accuracy  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()


import  pandas as pd
# Save history to CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv('Models/historyres50final.csv', index=False)


testing_data_path = 'dataset/newdata'
model_path = 'Models/epochscolab/modelResNe11.h5'

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



# load the model
model = load_model('renet50finalmodele.h5')

# set the batch size
batch_size = 32

# define image size
img_size = (224, 224)

# load the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/newdata',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

scores = model.evaluate(test_generator, verbose=1, steps=len(test_generator))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# make predictions on the test set
y_pred = model.predict_generator(test_generator, verbose=1)

# convert true labels to binary array
y_true = test_generator.classes
num_classes = test_generator.num_classes
y_true = np.eye(num_classes)[y_true]

# calculate the fpr and tpr for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plot the ROC curves for each class and the micro-average ROC curve
plt.figure()
lw = 2
colors = ['red', 'blue', 'green', 'orange', 'purple'] # add more colors if needed
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()

