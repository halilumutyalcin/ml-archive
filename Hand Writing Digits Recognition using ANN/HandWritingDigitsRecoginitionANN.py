import matplotlib.pyplot as plt
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Importing Dataset
from keras.datasets import mnist

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
# Normalizing Dataset
train_img = keras.utils.normalize(train_img, axis=1)
test_img = keras.utils.normalize(test_img, axis=1)

# Building Model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation="softmax"))

# Compiling Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Fitting the Model
model.fit(train_img, train_lab, epochs=10)

# Evaluate the Model
print(model.evaluate(test_img, test_lab))

# Predicting First 10 test images
pred = model.predict(test_img[:10])
# print(pred)
p = np.argmax(pred, axis=1)
print(p)
print(test_lab[:10])

# Visualizing prediction
for i in range(10):
    plt.imshow(test_img[i], cmap='binary')
    plt.title("Original: {}, Predicted: {}".format(test_lab[i], p[i]))
    plt.axis("Off")
    plt.figure()
    plt.show()
