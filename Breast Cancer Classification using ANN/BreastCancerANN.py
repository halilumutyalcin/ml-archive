import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

cancer = pd.read_csv("data.csv")
data = cancer.copy()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values

x_data = data.drop(["diagnosis"], axis=1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
y = y.reshape(y.shape[0], 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=42)


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    y_pred = classifier.predict(X_test)
    y_pred = y_pred > 0.5
    cm = confusion_matrix(Y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig("h.png")
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, epochs=100)


accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()

print("\nAccuracy mean: {:.2f}%".format(mean * 100))
print("Accuracy variance: {:.2f}%\n".format(variance * 100))

print("First Accuracy: {:.2f}%".format(accuracies[0] * 100))
print("Second Accuracy: {:.2f}%".format(accuracies[1] * 100))
print("Third Accuracy: {:.2f}%".format(accuracies[2] * 100))