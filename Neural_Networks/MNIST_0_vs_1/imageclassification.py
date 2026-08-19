import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Sequential
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 0) | (y_train == 1)
x_train = x_train[train_filter]
y_train = y_train[train_filter]
test_filter = (y_test == 0) | (y_test == 1)
x_test = x_test[test_filter]
y_test = y_test[test_filter]
x_train = x_train / 255.0
x_test = x_test / 255.0
model = Sequential([Flatten(input_shape=(28,28)),Dense(units=128,activation='relu'),Dense(units=1,activation='sigmoid')])
model.compile(loss = 'binary_crossentropy',optimizer ='adam',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=32)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
image = Image.open("ML\\1.png")
image = image.convert("L")
image = image.resize((28,28))
image_array = np.array(image)
image_array = 255 - image_array
image_array = image_array
image_input = image_array.reshape(1, 28, 28)
prediction = model.predict(image_input,verbose = 1)
print("Probability :",prediction[0][0])
predicted_class = 1 if prediction[0][0] >= 0.5 else 0
print("Predicted:", predicted_class)
plt.imshow(image_array, cmap="gray")
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()