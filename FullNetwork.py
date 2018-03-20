import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
from numpy.random import seed
from tensorflow import set_random_seed

class FNet:

    #self.input_size                    -> size of the input
    #self.output_size                   -> size of the output
    #self.layer                         -> layer of the full connected net

    def __init__(self, input_size, output_size, layer):
        self.input_size = input_size
        self.output_size = output_size
        self.layer = layer