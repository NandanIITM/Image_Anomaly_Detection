#import packages
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
from model import Model
from utils import Utils
import json
import warnings



#this variable need to be set before running the notebook
#data path
base_path = os.path.pardir

sandwich_path = os.path.join(base_path, 'sushi_or_sandwich_photos','sandwich')
sushi_path = os.path.join(base_path, 'sushi_or_sandwich_photos','sushi')
sandwich_model_path = os.path.join(base_path , 'model', 'sandwich_model')
sushi_model_path = os.path.join(base_path, 'model' , 'sushi_model')
serve_image_path = os.path.join(base_path, 'sushi_or_sandwich_photos','all_images')
save_image_path = os.path.join(base_path, 'predictions')


#image size after compression
compress_image_size = 256







if __name__=='__main__':

  #read sandwich images
  obj = Utils(compress_image_size)
  sandwich_images, original_sandwich_index = obj.read_images(sandwich_path)
  #split dataset into train and test
  train_split=0.8
  sandwich_train, sandwich_test, sandwich_train_images_idx, sandwich_test_images_idx = obj.split_dataset(sandwich_images, original_sandwich_index, train_split)
  

  #read sushi images
  sushi_images, original_sushi_index = obj.read_images(sushi_path)
  #split dataset into train and test
  sushi_train, sushi_test, sushi_train_images_idx, sushi_test_images_idx = obj.split_dataset(sushi_images, original_sushi_index, train_split)

  #anomaly threshold dict
  thresholds = {}

  #parameters
  kernel_size = (3, 3)
  filters = (32, 16, 8)
  activation ='relu'
  optimizer = 'adam'
  loss = 'binary_crossentropy'
  maxpoolsize = (2, 2)
  epochs = 15
  batch_size = 16

  #build sandwich training model
  sandwich_obj = Model(sandwich_train, kernel_size, filters, activation, optimizer, loss, maxpoolsize)
  sandwich_autoencoder = sandwich_obj.buildConvAutoEncoder()
  sandwich_history = sandwich_autoencoder.fit(sandwich_train, sandwich_train,
                epochs= epochs,
                batch_size= batch_size, validation_data=(sandwich_test, sandwich_test)
                )

  plots_path = os.path.join(base_path, 'plots')
  obj.plotReconstruction(sandwich_autoencoder, sandwich_train, plots_path, id='sandwich')
  obj.plotLossCurve(sandwich_history, plots_path, id='sandwich')
  sandwich_threshold = sandwich_obj.find_threshold(sandwich_autoencoder, sandwich_train)
  #print(f"Threshold: {sandwich_threshold}")
  thresholds['sandwich_threshold'] = sandwich_threshold
  sandwich_autoencoder.save(sandwich_model_path)


  #predictions on sandwich test_dataset
  sandwich_predictions = sandwich_obj.get_predictions(sandwich_autoencoder, sandwich_images, sandwich_threshold)
  sandwich_anomaly_count = Counter(sandwich_predictions)
  print(f'Displaying {sandwich_anomaly_count[0.0]} probable sushidi from sandwich model')
  obj.show(sandwich_predictions, sandwich_path, original_sandwich_index, save_image_path,  id='sandwich')
  

  
  #build sushi training model
  sushi_obj = Model(sushi_train, kernel_size, filters, activation, optimizer, loss, maxpoolsize)
  sushi_autoencoder = sushi_obj.buildConvAutoEncoder()
  sushi_history = sushi_autoencoder.fit(sushi_train, sushi_train,
                epochs= epochs,
                batch_size= batch_size, validation_data=(sushi_test, sushi_test)
                )
  obj.plotReconstruction(sushi_autoencoder, sushi_train,  plots_path, id='sushi')
  obj.plotLossCurve(sushi_history, plots_path, id='sushi')

  sushi_threshold = sushi_obj.find_threshold(sushi_autoencoder, sushi_train)
  #print(f"Threshold: {sushi_threshold}")
  thresholds['sushi_threshold'] = sushi_threshold


  sushi_autoencoder.save(sushi_model_path)


  #predictions on sushi test_dataset
  sushi_predictions = sushi_obj.get_predictions(sushi_autoencoder, sushi_images, sushi_threshold)
  sushi_anomaly_count = Counter(sushi_predictions)
  print(f'Displaying {sushi_anomaly_count[0.0]} probable sushidi from sushi model')
  obj.show(sushi_predictions, sushi_path, original_sushi_index, save_image_path, id='sushi')



