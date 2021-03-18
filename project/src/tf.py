import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from tensorflow import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

model = keras.Sequential([
    keras.layers.Dense(4100, input_dim=8200, activation="relu"),
    keras.layers.Dense(1025, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(6, activation="sigmoid")
])

# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Write the graph out to a file.
with open('graph.pb', 'wb') as f:
    f.write(tf.Graph().as_graph_def().SerializeToString())

print("reading")

read = 1024
max = 400000
skip = 100000
epochs = 200


def to_chars(str):
    chars = []
    l = len(str)
    for x in range(0, 4096):
        if x < l:
            chars.append(ord(str[x]))
        else:
            chars.append(0)
    return chars


inputs = []
classes = []
i = 0
acc = 0
pred = 0
for r in csv.reader(open("../assets/results.csv"), delimiter=','):
    if i < skip:
        i += 1
        continue

    data = []
    data.append(int(r[13]) > 0)
    data.extend([int(x) for x in r[14:21]])
    data.extend(to_chars(r[21]))
    data.extend(to_chars(r[22]))
    inputs.append(np.array(data))
    classes.append(int(r[12]))
    i += 1

    if i % read == 0:
        inputs = np.array(inputs)
        classes = np.array(classes)

        if i > max:
            outputs = model.predict(inputs, verbose=0)
            p = np.sum([classes[j] == o for j, o in enumerate(outputs)])/read
            print('\r[{}] Prediction Accuracy: {}/{}'.format(i, (p*100), (pred/((i-max)/read)*100)), end="")
            pred += p
        else:
            model.fit(inputs, classes, epochs=epochs, batch_size=read, verbose=0)

            # evaluate the keras model
            _, accuracy = model.evaluate(inputs, classes, verbose=0)
            print('\r[{}] Learn Accuracy: {}/{}'.format(i, (accuracy*100), (acc/(i/read)*100)), end="")
            acc += accuracy

        inputs = []
        classes = []

print("\nreaded")

print('\rTest Accuracy: %.2f' % (acc/(i/read)*100))
print('Prediction Accuracy: %.2f'%(pred/((i-max)/read)*100))
