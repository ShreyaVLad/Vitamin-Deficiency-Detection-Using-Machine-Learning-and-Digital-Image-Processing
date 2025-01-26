import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model

# Data preprocessing
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    r"C:\Users\Shreya\PycharmProjects\miniproject_DIP_04_09\Training",
    target_size=(224,224),
    batch_size=9,
    class_mode='categorical'
)
validation_dataset = validation.flow_from_directory(
    r"C:\Users\Shreya\PycharmProjects\miniproject_DIP_04_09\Validation",
    target_size=(224,224),
    batch_size=9,
    class_mode='categorical'
)

# Check the class indices
print(train_dataset.class_indices)

# Number of classes
num_classes = 5

# Model building
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Model compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model_fit = model.fit(
    train_dataset,
    steps_per_epoch=8,  # Adjust based on your dataset
    epochs=50,
    validation_data=validation_dataset
)

# Save the model
model.save('model.h5')

# Testing the model with an example image
img = image.load_img(r"C:\Users\Shreya\PycharmProjects\miniproject_DIP_04_09\Testing\Vitamin_detection\vitaminC(30).jpg", target_size=(224,224))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
predicted_class = model.predict(images)
val = np.argmax(predicted_class)

# Print the result based on the prediction
if val == 0:
    print("best")
elif val == 1:
    print("good")
elif val == 2:
    print("average")
elif val == 3:
    print("poor")
elif val == 4:
    print("very poor")

# Mapping the class indices to detect the vitamin
class_indices = train_dataset.class_indices
index_to_class = {v: k for k, v in class_indices.items()}  # Reverse the dictionary to map index to class

predicted_vitamin = index_to_class[val]
print(f"The predicted vitamin difficiency is : {predicted_vitamin}")

# Plotting the accuracy and loss
# Plotting training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model_fit.history['accuracy'], label='Train Accuracy')
plt.plot(model_fit.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(model_fit.history['loss'], label='Train Loss')
plt.plot(model_fit.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()
