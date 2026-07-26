# Covid Detection

The Covid Detection Project is a medical report analysis of Covid positive and Covid negative (Normal) categorical data. This project predicts Covid status of a patient based on their X-ray reports. The Dataset required for this project includes the images of X-ray's of patients. This Dataset consists of two categories: Covid dataset containing Covid Positive reports and Normal dataset containing Covid Negative reports. Algorithm I've used to design this project Convolutional Neural Network (CNN) which automatically learns patterns from images and classifies the images based on these patterns.

Step by Step designing of Covid Detection Project:

1. Data Collection: Collection of required data of X-Rays of Covid Patients containing Covid Positive & Negative (Normal).

2.	Data Split: Collected dataset is been split into two datasets – Train (Covid Det) & Test Dataset (Covid Test).

3.	Model Selection: The model built is of Sequential model to detect the Covid status based on the X-Ray images that has been used in our dataset and classifies it into two categories.
       1. Covid Positive
       2. Covid Negative (Normal)

4. Model Building: The Sequential model consists of different layers.
      1.	Convolutional Layer (Conv2D)
      2.	Max Pooling Layer (MaxPooling2D)
      3.	Batch Normalization Layer
      4.	Dropout Layer
      5.	Flatten Layer
      6.	Dense Layer 

   Layer-by-Layer Purpose:
   1. Convolution Layers (Conv2D):
           * Detect features like: edges, textures, shapes, etc 
           * Filters (32, 64, 96 …) = number of features learned
   2. Activation Function (ReLU):
           * Formula:
                 f(x) = max(0, x) 
           * Removes negative values 
           * Adds non-linearity (helps model learn complex patterns)
   3. MaxPooling (MaxPooling2D):
           * Reduces image size
           * Keeps only important features
           * Faster computation and Reduces overfitting
   4. Batch Normalization:
           * Normalizes data after each layer
           * Stabilizes learning and Speeds up training
           * Prevents large value fluctuations
   5. Dropout Layer:
           * Randomly turns off some neurons
           * Prevents overfitting
           * Improves generalization
   7. Flatten Layer:
           * Converts 2D feature maps → 1D vector
   8. Dense Layer (Fully Connected):
           * Learns final patterns for classification
           * 128 neurons → deeper understanding
   9. Output Layer (Dense(2, softmax)):
           * 2 neurons = 2 classes 
           * Softmax gives probabilities 


5. Model Compilation:
  * It prepares the model for training 

  Training Optimization Techniques:
  1. Optimizer = 'adam':
    * Controls how weights are updated
    * Uses smart learning technique (combines momentum + adaptive learning)
    * Faster training and Efficient convergence 
  
  2. Loss Function = 'categorical_crossentropy':
    * Measures how wrong the model is
    * Minimize loss → better predictions
  
  3. Metric = 'accuracy':
    * Tells how many predictions are correct 
    Formula:
      Accuracy = (Correct Predictions / Total Predictions)

6. Model Training:
  * It trains the neural network 
  * The model learns from data by adjusting weights
     
  1. Training Data (training_set):
    * Input images + labels
    * Model learns patterns from this data 
  2. Epochs = 50:
    * 1 epoch = full pass over training data
    * Model trains 50 times on the dataset 
  3. Steps per Epoch = 50:
    * Number of batches processed in one epoch 
  4. Validation Data (test_set):
    * Used to check performance 
    * Detect overfitting and Measures real-world performance 
  6. Validation Steps = 20:
    * Number of batches used for validation

7. Model Evaluation (Testing) & Prediction:
  1. Get Predictions:
      * Model takes test images 
      * Outputs probabilities for each class
  2. Convert Probabilities → Class Index:
    * Picks the highest probability 
    * Converts to class index (0 or 1) 
  3. Get Actual Classes:
    * These are the correct labels 
    * Provided by dataset 
  4. Get Class Labels (Names):
    * Maps index → label name 
  5. Convert Index → Label:
    * Converts numbers → readable labels 
