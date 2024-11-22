import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10
dropout_rate = 0.2
hidden_layer_size = 64

# build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(hidden_layer_size, activation='relu'),
    Dropout(dropout_rate),
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import time
start_time = time.perf_counter()

# train the model
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

end_time = time.perf_counter()
time_lapse = end_time - start_time
print(f"Training Time: {time_lapse} seconds")

# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest Accuracy:', test_acc)

# predict the label for a single image
sample_image = test_images[0:1] # use this for the prediction
show_image = test_images[0].squeeze() # use this for displaying the image that the model will classify

sample_label = test_labels[0]  # corresponding label for the image taken

plt.imshow(show_image, cmap='gray') # display the image in grayscale
plt.title(f"Label: {sample_label}") # add the label as the title
plt.axis('off') # hide the axis for clarity
plt.show()

prediction = model.predict(sample_image) # use our model to classify the image
predicted_label = np.argmax(prediction)
print(f"Predicted label: {predicted_label}")