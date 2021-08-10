#import packages
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
from model import Model
import json
import warnings
import skimage.io
import skimage.viewer
from skimage.io import imread, imshow



class Utils:


    def __init__(self, image_size):
      self.compress_image_size = image_size



    def read_images(self, path):
      '''
            Function to read images from a directory and return images in np matrix with index
            Args:
            path - directory path
            Return:
            images - np image matrix
            image_index - dict {key:index, value:'image_name'}

      '''

      data_size = len(os.listdir(path))
      new_size = (self.compress_image_size, self.compress_image_size)
      images = np.zeros((data_size, new_size[0], new_size[1], 1))
      image_index = {}
      for i, files in enumerate(os.listdir(path)):
        #converting image into grayscale
        image = Image.open(path+'/'+files).convert('L')
        image = image.resize(new_size)
        image_index[i] = files
        np_array = np.array(image)
        np_array = np_array.astype('float32') / 255.
        np_array = np.reshape(np_array, (new_size[0], new_size[1], 1))
        images[i] = np_array
      return images, image_index





    def plotReconstruction(self, model, images, plots_path, id):
      '''
            Function to plot original and reconstruct images
            Args:
            model - model obj, autoencodel model
            images - np matrix of images
            id - str, ('sushi', 'sandwich')
      
      '''

      decoded_imgs = model.predict(images)
      n = 10
      plt.figure(figsize=(20, 8))
      for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(images[i].reshape(self.compress_image_size, self.compress_image_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(self.compress_image_size, self.compress_image_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
      #plt.show()
      path = os.path.join(plots_path, id, 'reconstruction.jpeg')
      plt.savefig(path)
      plt.clf()





    def plotLossCurve(self, history, plots_path, id):
      '''
            Function to plot loss curve
            Args:
            history: model history obj
            id - str, ('sushi', 'sandwich')
      
      '''

      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.xlabel('Epochs')
      plt.ylabel('MSLE Loss')
      plt.legend(['loss', 'val_loss'])
      plt.title(id+ '_loss')
      #plt.show()
      path = os.path.join(plots_path, id, 'loss.jpeg')
      plt.savefig(path)
      




    def split_dataset(self, images, index, split_size=0.8):
      '''
            Function to split dataset into train and test 
            Args:
            images: np array of images
            index: dict, index of images
            split_size: float, e.g -(0.8)

            Returns:
            train_images : np array of train images
            test_images :np array of test images
            train_images_idx : dict, index of train images
            test_images_idx : dict, index of test images
      
        '''
      train_images_idx = {}
      test_images_idx = {}
      indices = np.random.permutation(images.shape[0])
      train_split = int(images.shape[0]*split_size)
      training_idx, test_idx = indices[:train_split], indices[train_split:]
      training, test = images[training_idx,:], images[test_idx,:]
      
      for i, tr_idx in enumerate(training_idx):
        train_images_idx[i] = tr_idx

      for i, te_idx in enumerate(test_idx):
        test_images_idx[i] = te_idx

      
      return training, test, train_images_idx, test_images_idx





    def show(self, predictions, prediction_path, original_sandwich_index, save_image_path,  id):
       '''
            Function to display anomaly images
            Args:
            predictions : np array of predictions
            index: dict, index of images
            prediction_path: path containing images
      
       '''
       for i, value in enumerate(predictions):
          if value==0.0:
            filename = original_sandwich_index[i]
            path = prediction_path+'/'+filename
            image = skimage.io.imread(path)
            plt.figure()
            plt.imshow(image)
            save_path = os.path.join(save_image_path, id, filename)
            plt.savefig(save_path)



