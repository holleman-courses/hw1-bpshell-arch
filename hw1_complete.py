#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

## 

def build_model1():
    model = Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),

        layers.Dense(128, activation='leaky_relu'),

        layers.Dense(128, activation='leaky_relu'),

        layers.Dense(128, activation='leaky_relu'),

        layers.Dense(10)  # logits, no activation
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model

def build_model2():
  model = Sequential([
      layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same",
                    activation="relu",input_shape=(32, 32, 3)),
      layers.BatchNormalization(),

      layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same",
                    activation="relu",),
      layers.BatchNormalization(),

      layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu",),
      layers.BatchNormalization(),

      layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.Flatten(),
      layers.Dense(10)
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])

  return model

def build_model3():
  model = Sequential([
      layers.SeparableConv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same",
                    activation="relu",input_shape=(32, 32, 3)),
      layers.BatchNormalization(),

      layers.SeparableConv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same",
                    activation="relu",),
      layers.BatchNormalization(),

      layers.SeparableConv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu",),
      layers.BatchNormalization(),

      layers.SeparableConv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.SeparableConv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.SeparableConv2D(128, kernel_size=(3, 3), padding="same",
                    activation="relu", ),
      layers.BatchNormalization(),

      layers.Flatten(),
      layers.Dense(10)
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = Sequential([
      layers.Conv2D(16, kernel_size=(3, 3), padding="same",
                    activation="relu",input_shape=(32, 32, 3)),
      layers.MaxPooling2D(pool_size=(2, 2)),

      #layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same"),

      layers.Conv2D(36, kernel_size=(3, 3), activation="relu", padding="same"),

      layers.MaxPooling2D(pool_size=(2, 2)),

      layers.DepthwiseConv2D(kernel_size=(2, 2), padding="valid"),

      layers.Conv2D(36, kernel_size=(1, 1), activation="relu", padding="valid"),
      layers.Conv2D(36, kernel_size=(1, 1), activation="relu", padding="valid"),

      layers.Flatten(),
      layers.Dense(16, activation='relu'),
      layers.Dense(10)
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])

  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

  # Normalize pixel values to [0, 1]
  train_images = train_images.astype("float32") / 255.0
  test_images = test_images.astype("float32") / 255.0

  # Split training data into training and validation sets
  val_images = train_images[-5000:]
  val_labels = train_labels[-5000:]

  train_images = train_images[:-5000]
  train_labels = train_labels[:-5000]

  ########################################
  ## Build and train model 1

  model1 = build_model1()
  model1.summary()

  # compile and train model 1.
  train_hist = model1.fit(
      train_images,
      train_labels,
      validation_data=(val_images, val_labels),
      epochs=60)

  test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)

  ## Build, compile, and train model 2 (DS Convolutions)

  model2 = build_model2()
  model2.summary()

  # compile and train model 2.
  train_hist2 = model2.fit(
      train_images,
      train_labels,
      validation_data=(val_images, val_labels),
      epochs=30)

  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy Model 2:', test_acc2)

  img = np.array(
      keras.utils.load_img(
          "test_image_dog.png",  # <-- change if needed
          color_mode="rgb",
          target_size=(32, 32)
      )
  )

  img = img.astype("float32") / 255.0

  img = np.expand_dims(img, axis=0)

  # Run model prediction
  logits = model2(img)
  predicted_class = np.argmax(logits, axis=1)[0]

  print("Predicted class:", class_names[predicted_class])

  ## Build and train model 3

  model3 = build_model3()
  model3.summary()

  # compile and train model 3.
  train_hist3 = model3.fit(
      train_images,
      train_labels,
      validation_data=(val_images, val_labels),
      epochs=30)

  test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy Model 3:', test_acc3)

  ### Repeat for model 3 and your best sub-50k params model

  model50k = build_model50k()
  model50k.summary()

  checkpoint = keras.callbacks.ModelCheckpoint(
      "best_model.h5",
      monitor="val_accuracy",
      save_best_only=True,
      verbose=1
  )

  # compile and train model 50k

  train_hist4 = model50k.fit(
      train_images,
      train_labels,
      validation_data=(val_images, val_labels),
      epochs=30,
      callbacks=[checkpoint]
  )

  test_loss4, test_acc4 = model50k.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy Model 4:', test_acc4)


