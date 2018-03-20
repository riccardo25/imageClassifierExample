import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
from numpy.random import seed
from tensorflow import set_random_seed

class CFilter:

    #self.filter_size                   -> size of the filter (3x3, 5x5)
    #self.num_filters                   -> number of convolutional filters (32, 64)
    #self.layer                         -> layer of the convolutional filter

    def __init__(self, filter_size, num_filters, layer):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.layer = layer