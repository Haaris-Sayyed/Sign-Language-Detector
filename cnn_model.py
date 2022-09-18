# Part 2 - Building the CNN

# --------------------------------------------------
# importing the Keras libraries and packages
# --------------------------------------------------

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from tensorflow.keras import optimizers

# --------------------------------------------------
# Initialing the CNN
# --------------------------------------------------
classifier = Sequential()

# --------------------------------------------------
# Step 1 - Convolutio Layer
# --------------------------------------------------
classifier.add(Convolution2D(
    32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# --------------------------------------------------
# step 2 - Pooling
# --------------------------------------------------
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# --------------------------------------------------
# Adding second convolution layer
# --------------------------------------------------
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# --------------------------------------------------
# Adding 3rd Concolution Layer
# --------------------------------------------------
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# --------------------------------------------------
# Step 3 - Flattening
# --------------------------------------------------
classifier.add(Flatten())

# --------------------------------------------------
# Step 4 - Full Connection
# --------------------------------------------------
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation='softmax'))

# --------------------------------------------------
# Compiling The CNN
# --------------------------------------------------
classifier.compile(
    optimizer=optimizers.SGD(lr=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# --------------------------------------------------
# Part 2 Fittting the CNN to the image
# --------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'Dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

model = classifier.fit_generator(
    training_set,
    steps_per_epoch=800,
    epochs=25,
    validation_data=test_set,
    validation_steps=6500
)

# --------------------------------------------------
# Saving the model
# --------------------------------------------------
classifier.save('Trained_model.h5')

print(model.history.keys())

# --------------------------------------------------
# summarize history for accuracy
# --------------------------------------------------
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# --------------------------------------------------
# summarize history for loss
# --------------------------------------------------

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
