# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:15:43 2018

@author: Tomas
"""

import NotMNIST_Functions_Homework as nmnist
import os
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

"""
First, we'll download the dataset to our local machine. The data consists of 
characters rendered in a variety of fonts on a 28x28 image. The labels are 
limited to 'A' through 'J' (10 classes). The training set has about 500k and 
the testset 19000 labeled examples. Given these sizes, it should be possible 
to train models quickly on any machine.
"""
train_filename = nmnist.maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = nmnist.maybe_download('notMNIST_small.tar.gz', 8458043)

"""
Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labeled A through J.
"""
train_folders = nmnist.maybe_extract(train_filename)
test_folders = nmnist.maybe_extract(test_filename)

"""
Problem 1
Let's take a peek at some of the data to make sure it looks sensible. Each 
exemplar should be an image of a character A through J rendered in a different 
font. Display a sample of the images that we just downloaded. Hint: you can use 
the package IPython.display.
"""
#nmnist.displaySampleOfImagesDataset()

"""
Now let's load the data in a more manageable format. Since, depending on your 
computer setup you might not be able to fit it all in memory, we'll load each 
class into a separate dataset, store them on disk and curate them independently. 
Later we'll merge them into a single dataset of manageable size.

We'll convert the entire dataset into a 3D array (image index, x, y) of floating 
point values, normalized to have approximately zero mean and standard deviation ~0.5 
to make training easier down the road.

A few images might not be readable, we'll just skip them.
"""
train_datasets = nmnist.maybe_pickle(train_folders, 45000)
test_datasets = nmnist.maybe_pickle(test_folders, 1800)

"""
Problem 2
Let's verify that the data still looks good. Displaying a sample of the 
labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
"""
for filename in train_datasets:
    nmnist.test_pickled_data(filename, 1)

root = os.path.splitext(os.path.splitext(test_filename)[0])[0]
for filename in test_datasets:
    nmnist.test_pickled_data(filename, 1)

"""
Problem 3
Another check: we expect the data to be balanced across classes. Verify that.
Merge and prune the training data as needed. Depending on your computer setup, 
you might not be able to fit it all in memory, and you can tune train_size as 
needed. The labels will be stored into a separate array of integers 0 through 9.

Also create a validation dataset for hyperparameter tuning.
"""
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = nmnist.merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = nmnist.merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


