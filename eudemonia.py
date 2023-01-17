# SC 1/17/2023 12:11 AM Prepare dataset by splitting into training, validation, and testsets, and preprocess the image in python.

import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2

# Define the paths to the dataset
data_path = 'C:\Users\shaqc\OneDrive\Documents\datasets\fer-2013'
train_path = 'C:\Users\shaqc\OneDrive\Documents\datasets\fer-2013\train'
test_path = 'C:\Users\shaqc\OneDrive\Documents\datasets\fer-2013\test'


# Create empty lists to store the images and labels
X_train = []
y_train = []
X_test = []
y_test = []

#Create a dictionary to map emotions to integer labels
emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Loop through the train subfolders and get the images and labels
for emotion in os.listdir(train_path):
    emotion_path = os.path.join(train_path, emotion)
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        image = cv2.resize(image, (48, 48)) # resize the image
        image = image / 255.0 # normalize the image
        X_train.append(image)
        y_train.append(emotion_dict[emotion])

# Loop through the test subfolders and get the images and labels
for emotion in os.listdir(test_path):
    emotion_path = os.path.join(test_path, emotion)
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        image = cv2.resize(image, (48, 48)) # resize the image
        image = image / 255.0 # normalize the image
        X_test.append(image)
        y_test.append(emotion_dict[emotion])

# Convert the lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

#Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split (X_train, y_train, test_size=0.2, random_state=42)

#Create data generators for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator()

#Fit the data generators on the training data
# train_datagen.fit(X_train)
# val_datagen.fit(X_val)

# Now you can use the X_train, X_val, X_test, y_train, y_val, y_test and data generators for training and evaluating your model.
# Note that this is just an example, you may need to adapt it to your specific use case.
# For example, you can use one-hot-encoding for the labels, or use a different data split ratio for the validation set.




from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a new dense layer for binary classification
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create a new model with the new dense layer
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# make predictions on new images
predictions = model.predict(new_images)
# predictions will be an array of probabilities for each class
# you can useargmax to get the index of the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

emotions = ['happy', 'sad', 'neutral', 'angry', 'disgusted', 'surprised']

# use the predicted class to look up the corresponding emotion label
predicted_emotions = [emotions[i] for i in predicted_class]

######Server Side SC 1/17/2023 12:02 AM
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('emotion_model.h5')
emotions = ['happy', 'sad', 'neutral', 'angry', 'disgusted', 'surprised']

@app.route('/predict', methods=['POST'])
def predict():
    # get the image data from the request
    image_data = request.files['image'].read()

    # use the TensorFlow model to predict the emotion
    predictions = model.predict(image_data)
    predicted_class = np.argmax(predictions, axis=1)
    emotion = emotions[predicted_class]

    # return the predicted emotion as json
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)

