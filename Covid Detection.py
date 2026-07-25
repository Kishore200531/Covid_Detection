from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./ 255)

training_set = train_datagen.flow_from_directory('C:\\Users\\U  KISHORE\\OneDrive\\ドキュメント\\Artificial Intelligence Internship\\Covid Det\\',
                                                 target_size=(128, 128), batch_size=8, class_mode='categorical', color_mode='grayscale')

labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory('C:\\Users\\U  KISHORE\\OneDrive\\ドキュメント\\Artificial Intelligence Internship\\Covid Test\\',
                                            target_size=(128, 128), batch_size=8, class_mode='categorical', color_mode='grayscale')

labels2 = (test_set.class_indices)
print(labels2)

#Model Training
model.fit(training_set,steps_per_epoch=50, epochs=50, validation_data=test_set, validation_steps=20)

#Making new predictions
import numpy as np
import pandas as pd

# Get predictions
predictions = model.predict(test_set)

# Get predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get actual class indices
true_classes = test_set.classes

# Get class labels
class_labels = list(test_set.class_indices.keys())

# Convert numbers to labels
predicted_labels = [class_labels[i] for i in predicted_classes]
true_labels = [class_labels[i] for i in true_classes]

# Create readable output
results = pd.DataFrame({
    "Image": test_set.filenames,
    "Actual": true_labels,
    "Predicted": predicted_labels
})

print(results)

results.to_csv("covid_test_results.csv", index=False)
print("Results saved to covid_test_results.csv")
