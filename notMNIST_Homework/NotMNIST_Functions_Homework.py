# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:29:51 2018

@author: Tomas
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from IPython.display import Image
import random

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = os.path.join(".", "datasets") # Change me to store data elsewhere
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

"""----------------------------------------------------------------------"""

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent

"""----------------------------------------------------------------------"""

def maybe_download(filename, expected_bytes, force=False):
  dest_filename = os.path.join(data_root, filename)
  if not os.path.exists(dest_filename):
    os.mkdir(dest_filename)
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename


"""----------------------------------------------------------------------"""

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  print ("Extracting: ", filename)
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  print("Extracting to:", root)
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  # Lists 
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
  ]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

"""----------------------------------------------------------------------"""

""" 
Display a sample of the images that we just downloaded. 
"""
def displaySampleOfImagesDataset():
    train_filename='.\\notMNIST_large.tar.gz'
    root = os.path.splitext(os.path.splitext(train_filename)[0])[0]
    
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    
    print (data_folders)
    
    counter = 0
    for folderPath in data_folders:
        filesInFolder = os.listdir(folderPath);
        for filename in filesInFolder: 
            if counter >= 10:
                break
            imagePath = os.path.join(folderPath, filename)
            print (imagePath)
            display(Image(filename=imagePath))
            counter += 1
    pass

"""----------------------------------------------------------------------"""

"""
Now let's load the data in a more manageable format. Since, depending on your computer 
setup you might not be able to fit it all in memory, we'll load each class into a separate 
dataset, store them on disk and curate them independently. Later we'll merge them into a 
single dataset of manageable size.
We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, 
normalized to have approximately zero mean and standard deviation ~0.5 to make training easier 
down the road. 
A few images might not be readable, we'll just skip them.
"""

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Full dataset mean:', np.mean(dataset))
  print('Full dataset Standard deviation:', np.std(dataset))
  return dataset

"""----------------------------------------------------------------------"""

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

"""----------------------------------------------------------------------"""

def test_pickled_data(filePath,numberOfImagesToPlot):
    with (open(filePath, "rb")) as openfile:
        while True:
            try:
                image = pickle.load(openfile)
                for i in range(numberOfImagesToPlot):
                    randomSampleIndex = random.randint(0, image.shape[0]) - 1;
                    plt.matshow(image[randomSampleIndex, :,:])
            except EOFError:
                break

"""----------------------------------------------------------------------"""

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

"""----------------------------------------------------------------------"""

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels

"""----------------------------------------------------------------------"""




