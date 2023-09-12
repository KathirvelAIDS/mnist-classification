# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit categorization of scanned handwriting images, together with answer verification. There are a number of handwritten digits in the MNIST dataset. The assignment is to place a handwritten digit picture into one of ten classes that correspond to integer values from 0 to 9, inclusively. The dataset consists of 60,000 handwritten digits that are each 28 by 28 pixels in size. In this case, we construct a convolutional neural network model that can categorise to the relevant numerical value.

## Neural Network Model
![i6](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/a0d12398-f269-4f63-96d7-b410db484ec0)



## DESIGN STEPS
```
STEP 1: Import the required packages
STEP 2: Load the dataset
STEP 3: Scale the dataset
STEP 4: Use the one-hot encoder
STEP 5: Create the model
STEP 6: Compile the model
STEP 7: Fit the model
STEP 8: Make prediction with test data and with an external data
```
## PROGRAM
```
Developed By: KATHIRVEL.A
Reg No: 212221230047
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape

X_test.shape

single_image= X_train[0]
     
single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
     
model.fit(X_train_scaled ,y_train_onehot, epochs=8,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

# Prediction for a single input
img = image.load_img('5.png')
type(img)

img = image.load_img('5.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![i1](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/1f8bc9fe-b8da-4ab9-9329-d8dd6fbe6ca2)

![i2](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/9d668398-a178-49eb-bae1-db6f6c5c34f0)


### Classification Report
![i3](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/f32b7b27-512a-4958-9816-db2dbc5f334f)


### Confusion Matrix

![i4](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/57f04f56-ca95-4853-9122-454d324ee791)


### New Sample Data Prediction

![i5](https://github.com/Hemapriya-2004/mnist-classification/assets/94184828/c41c5fa7-ea26-4a48-bf1c-e7dc467cd5b6)


## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
