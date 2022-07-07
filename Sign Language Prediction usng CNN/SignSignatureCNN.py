import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.optimizers import adam_v2
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from warnings import filterwarnings

filterwarnings('ignore')

x = np.load("dataset/X.npy")
y = np.load("dataset/Y.npy")

img_size = 64

x = x.reshape(-1, 64, 64, 1)
number_of_classes = y.shape[1]

list_y = [np.where(i == 1)[0][0] for i in y]
count = pd.Series(list_y).value_counts()

X_organized = np.concatenate((x[204:409, :],
                              x[822:1028, :],
                              x[1649:1855, :],
                              x[1443:1649, :],
                              x[1236:1443, :],
                              x[1855:2062, :],
                              x[615:822, :],
                              x[409:615, :],
                              x[1028:1236, :],
                              x[0:204, :]), axis=0)

x_train, x_test, y_train, y_test = train_test_split(X_organized, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(9, 9), padding='Same', activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPool2D(pool_size=(5, 5)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(7, 7), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(4, 4), strides=(3, 3)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation='softmax'))

optimizer = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(zoom_range=0.5, rotation_range=45)
datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size=250), epochs=100, validation_data=(x_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_predict = model.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true, y_predict_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
