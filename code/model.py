#import packages
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd


class Model:


  def __init__(self, data, kernel_size, filters, activation, optimizer, loss, maxpoolsize):
    '''
        Args:
        data : training data np-matrix
        kernel_size: tuple of 2 integers, specifying the height and width of the 2D convolution window
        filters: integer, the dimensionality of the output space
        activation: string, Activation function to use, e.g-'relu', 'sigmoid'
        optimizer: string, optimizer to use, e.g - 'adam', 'sgd'
        loss: string, loss function to use, e.g- 'binary_crossentropy'
        maxpoolsize: tuple of 2 integers, specifying the height and width of the maxpool

    '''
    self.data = data
    self.kernel_size = kernel_size
    self.filters = filters
    self.activation = activation
    self.optimizer = optimizer
    self.loss = loss
    self.maxpoolsize = maxpoolsize


  def buildConvAutoEncoder(self):
      input_img = keras.Input(shape=(self.data.shape[1], self.data.shape[2], self.data.shape[3]))
      x = layers.Conv2D(self.filters[0], self.kernel_size, activation= self.activation, padding= 'same')(input_img)
      x = layers.MaxPooling2D((self.maxpoolsize), padding='same')(x)
      x = layers.Conv2D(self.filters[1], self.kernel_size, activation= self.activation, padding= 'same')(x)
      x = layers.MaxPooling2D((self.maxpoolsize), padding= 'same')(x)
      x = layers.Conv2D(self.filters[2], self.kernel_size, activation= self.activation, padding='same')(x)
      encoded = layers.MaxPooling2D((self.maxpoolsize), padding= 'same')(x)
      
      # at this point the representation is (4, 4, 8) i.e. 128-dimensional
      
      x = layers.Conv2D(self.filters[2],  self.kernel_size, activation= self.activation, padding='same')(encoded)
      x = layers.UpSampling2D(self.maxpoolsize)(x)
      x = layers.Conv2D(self.filters[1], self.kernel_size, activation= self.activation, padding='same')(x)
      x = layers.UpSampling2D(self.maxpoolsize)(x)
      x = layers.Conv2D(self.filters[0], self.kernel_size, activation= self.activation, padding='same')(x)
      x = layers.UpSampling2D(self.maxpoolsize)(x)
      decoded = layers.Conv2D(1, self.kernel_size, activation= 'sigmoid', padding='same')(x)
      
      autoencoder = keras.Model(input_img, decoded)
      autoencoder.compile(optimizer= self.optimizer, loss=self.loss)
      autoencoder.summary()
      return autoencoder


  def get_predictions(self, model, test_data, threshold):
      '''
          Function to get prediction from model 
          Args:
          model : model obj, autoencodel model
          test_data: np image matrix for test data
          threshold: float

          Returns:
          preds: np array, output labels
    
      '''
      predictions = model.predict(test_data)
      # provides losses of individual instances
      test_data_flat = np.reshape(test_data, (test_data.shape[0],(test_data.shape[1]*test_data.shape[1])))
      prediction_flat = np.reshape(predictions, (test_data_flat.shape[0], test_data_flat.shape[1]))
      errors = tf.keras.losses.msle(prediction_flat, test_data_flat)
      # 0 = anomaly, 1 = normal
      anomaly_mask = pd.Series(errors) > threshold
      preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
      return preds



  def find_threshold(self, model, train_data):
      '''
            Function to find error threshold above which image will be classified as anomaly
            Args:
            model : model obj, autoencodel model
            train_data: np image matrix

            Returns:
            threshold: float
      
      '''
      reconstructions = model.predict(train_data)
      # provides losses of individual instances
      train_data_flat = np.reshape(train_data, (train_data.shape[0],(train_data.shape[1]*train_data.shape[1])))
      reconstruction_flat = np.reshape(reconstructions, (train_data_flat.shape[0], train_data_flat.shape[1]))
      reconstruction_errors = tf.keras.losses.msle(reconstruction_flat, train_data_flat)
      # threshold for anomaly scores
      threshold = np.mean(reconstruction_errors.numpy()) \
          + np.std(reconstruction_errors.numpy())
      return threshold


  def serving(self, image_path, sushi_model_path, sandwich_model_path, sandwich_threshold, sushi_threshold):
      '''
            Function to perform serving after training is complete

            Args:
            image_path : path of the serving  images 
            sushi_model_path : local model path of sushi_model
            sandwich_model_path : local model path of sandwich_model
            sandwich_threshold : error threshold above which image will be classified as anomaly for sandwich_model
            sushi_threshold :  error threshold above which image will be classified as anomaly for sushi_model

            Returns:
            anomalies : list, anomaly images names
      
      '''
      image_matrix, image_index = read_images(image_path)
      sandwich_model = keras.models.load_model(sandwich_model_path)
      sushi_model = keras.models.load_model(sushi_model_path)
      sandwich_predictions = get_predictions(sandwich_model, image_matrix, sandwich_threshold)
      sushi_predictions = get_predictions(sushi_model, image_matrix, sushi_threshold)
      anomalies=[]
      for i in range(len(sushi_predictions)):
        #anomaly
        if sushi_predictions[i]==0.0 and sandwich_predictions[i]==0.0:
          anomalies.append(image_index[i])

      return anomalies