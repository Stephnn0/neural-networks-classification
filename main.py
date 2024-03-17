import tensorflow as tf
import tensorflow_datasets as tfds

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

training_data, testing_data = data['train'], data['test']

name_classes = metadata.features['label'].names

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

training_data = training_data.map(normalize)
testing_data = testing_data.map(normalize)

training_data = training_data.cache()
testing_data = testing_data.cache()


for image, label in training_data.take(1):
  break
image = image.numpy().reshape((28,28))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(training_data.take(25)):
  image = image.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap=plt.cm.binary)
  plt.xlabel(name_classes[label])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_training_data = metadata.splits["train"].num_examples
num_test_data = metadata.splits["test"].num_examples

BATCH = 32

training_data = training_data.repeat().shuffle(num_training_data).batch(BATCH)
testing_data = testing_data.batch(BATCH)

import math

res = model.fit(training_data, epochs=5, steps_per_epoch= math.ceil(num_training_data / BATCH))

plt.xlabel("# epoch")
plt.ylabel("loss")
plt.plot(res.history["loss"])

import matplotlib.pyplot as plt

index = 50

for image, label in testing_data.take(index): 
    image = image[0].numpy() 
    image = image.reshape((28, 28)) 

    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()   

    # Perform prediction
    predictions = model.predict(image.reshape(1, 28, 28))   
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

    class_names = metadata.features['label'].names

    # Display prediction result
    print(f"Predicted Class: {class_names[predicted_class]}")