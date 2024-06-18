import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt

# broj klasa i oblik ulaznih podataka
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# skaliranje podataka na raspon od 0 do 1
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# dodavanje dimenzije za kanale
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# one hot coding
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# definiranje modela
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1), name="ulaz"))
model.add(layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax', name="izlaz"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# treniranje modela
model.fit(x_train_s, y_train_s, epochs=5, batch_size=32)

# evaluacija modela na testnom skupu
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

model.save("mymodel")
