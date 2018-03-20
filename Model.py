import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
import json
from numpy.random import seed
from tensorflow import set_random_seed
from ConvolutionalFilter import CFilter
from FullNetwork import FNet


#Model of Convolutional NN to elaborate Images
class ModelCNN():

    #FIELDS
    # self.session                  -> model session
    # self.accuracy                 -> accuracy model
    # self.x                        -> input of the model (placeholder)
    # self.y_true                   -> output (true values) of the model to compare with the predicted (placeholder)
    # self.y_true_class             -> class of true value of the output
    # self.convolutionalLayers      -> layers of convolutional filter of the model
    # self.layer_flat               -> flat layer of the model
    # self.fullConnectedLayers      -> layers of full connected net
    # self.y_pred                   -> predicted output
    # self.y_pred_class             -> predicted class
    # self.cross_entropy            -> entropy of the model
    # self.cost                     -> final cost of the model
    # self.optimizer                -> optimizer for back propagation
    # self.correct_prediction       -> prediction 
    # self.batch_size               -> number of images to train per time
    # self.modelPath                -> path of the model to load-save
    # self.modelName                -> name of the model
    # self.classes                  -> classes of model to split data
    # self.nClasses                 -> number of classes 
    # self.imgSize                  -> image to resize the image
    # self.imgNumChannels           -> mumber of channels of images

    def __init__(self):

        self.__getImgSize__()
        self.__getImgNumChannels__()
        self.__getBatchSize__()
        self.__getModelPath__()
        self.__getNumClasses__()

        data = json.load(open(".\\config.json"))
        #MODEL CREATION
        #creation of the session
        self.session = tf.Session()
        #input of the network
        self.x = tf.placeholder(tf.float32, shape=[None, self.imgSize, self.imgSize, self.imgNumChannels], name='x')
        #labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.nClasses], name='y_true')
        self.y_true_class = tf.argmax(self.y_true, dimension=1)
        #creations of convolutional layered filter
        self.convolutionalLayers = []

        for cf in data["convolution_filters"]:
            
            inputlayer = self.x
            num_input_ch = self.imgNumChannels
            if(len(self.convolutionalLayers) != 0):
                inputlayer = self.convolutionalLayers[len(self.convolutionalLayers)-1].layer
                num_input_ch = self.convolutionalLayers[len(self.convolutionalLayers)-1].num_filters
            filtersize = cf["filter_size"]
            nfilters = cf["num_filters"]
            newCL = self.__create_convolutional_layer__(input = inputlayer, num_input_channels = num_input_ch, conv_filter_size = filtersize, num_filters = nfilters)
            self.convolutionalLayers.append( CFilter(filtersize, nfilters, newCL ))
        #creation flat layer
        self.layer_flat = self.__create_flatten_layer__(self.convolutionalLayers[len(self.convolutionalLayers)-1].layer)
        #creation of full connected layers
        self.fullConnectedLayers = []

        if(data["full_connected"][len(data["full_connected"])-1]["output_size"] != self.nClasses):
            raise ValueError('In config.json, number of classes must be equal of the last full connected output layer variables!')

        for fc in data["full_connected"]:
            inputlayer = self.layer_flat
            num_input = self.layer_flat.get_shape()[1:4].num_elements()
            if(len(self.fullConnectedLayers) != 0):
                inputlayer = self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer
                num_input = self.fullConnectedLayers[len(self.fullConnectedLayers)-1].output_size
            num_output = fc["output_size"]
            use_re = fc["use_relu"]

            newFC = self.__create_fc_layer__(input = inputlayer, num_inputs = num_input, num_outputs = num_output, use_relu = use_re )
            
            self.fullConnectedLayers.append( FNet(input_size = num_input, output_size = num_output, layer = newFC))
        
        
        #predictions
        self.y_pred = tf.nn.softmax(self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer, name='y_pred') #standard use of softmax -> 0 - 1 probability spread to the output
        self.y_pred_class = tf.argmax(self.y_pred, dimension=1) #return the index of the max value output
        
        #initialize variables
        self.session.run(tf.global_variables_initializer())

        #set the minimization cost like cross-entropy (ok not the normal norm)
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = data["learning_rate"]).minimize(self.cost)
        self.correct_prediction = tf.equal(self.y_pred_class, self.y_true_class)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.session.run(tf.global_variables_initializer()) 

    ### reads the batch size from config.json
    def __getBatchSize__(self):
        data = json.load(open(".\\config.json"))
        self.batchSize = data["batch_size"]

    ### get number of classes
    def __getNumClasses__(self):
        data = json.load(open(".\\config.json"))
        
        self.classes = data["classes"]
        self.nClasses = len(data["classes"])

    ### reads the path to save-load the model from config.json
    def __getModelPath__(self):
        data = json.load(open(".\\config.json"))
        self.modelPath = data["model_path"]
        self.modelName = data["model_name"]


    ### reads the imagesize
    def __getImgSize__(self):

        data = json.load(open(".\\config.json"))
        self.imgSize = data["img_size"]

    ### reads the image channels
    def __getImgNumChannels__(self):

        data = json.load(open(".\\config.json"))
        self.imgNumChannels = data["img_num_channels"]

    ### creates an array of variables to weights 
    def __create_weights__(self, shape):

        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    ### creates an array of variables to bias
    def __create_biases__(self, size):

        return tf.Variable(tf.constant(0.05, shape=[size]))

    ### create a convolutional layer (filter) with variable weights and biases 
    def __create_convolutional_layer__(self, input, num_input_channels, conv_filter_size, num_filters):  

        ## We shall define the weights that will be trained using create_weights function.
        weights = self.__create_weights__(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        ## We create biases using the create_biases function. These are also trained.
        biases = self.__create_biases__(num_filters)
        ## Creating the convolutional layer, strides is the number of sliding window
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        layer += biases
        ## We shall be using max-pooling.  
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)

        return layer

    ### create a layer that serialize the values
    def __create_flatten_layer__(self, layer):
        
        #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()
        ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements() #<- multiply
        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features]) #creates the stream

        return layer

    ### create the full connected layer
    def __create_fc_layer__(self, input, num_inputs, num_outputs, use_relu=True):

        #Let's define trainable weights and biases.
        weights = self.__create_weights__(shape=[num_inputs, num_outputs])
        biases = self.__create_biases__(num_outputs)
        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.add(tf.matmul(input, weights) , biases)
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    ### show the progress of training
    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        #Calculate the accuracy of training data
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        #Calculate the accuracy of validation data
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)

        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f},  Cost: {3:.3f}"

        print(msg.format(epoch + 1, acc, val_acc, val_loss, self.cost))

    ### trains the model and saves it into the pre defined location
    def train(self, iterations, data_train, label_train, data_eval, label_eval):
        
        saver = tf.train.Saver()
        for i in range(0, iterations):
            index = int(i % self.batchSize) * int(len(data_train)/self.batchSize)
            x_batch = data_train[index: index+self.batchSize]
            y_true_batch = label_train[index: index+self.batchSize]
            x_valid_batch = data_eval
            y_valid_batch = label_eval
            #feeding
            feed_dict_tr = {self.x: x_batch, self.y_true: y_true_batch}
            feed_dict_val = {self.x: x_valid_batch, self.y_true: y_valid_batch}
            #running
            self.session.run(self.optimizer, feed_dict=feed_dict_tr)
            
            if i % 10 == 0: 
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / 10)    
                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(self.session, self.modelPath+self.modelName) 

            if i == iterations-1:
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / 10)    
                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(self.session, self.modelPath+self.modelName)
    
    ### evaluates data from the model, and load it from disk if needed
    def evaluate(self, data_in, load_model = False):

        if ( load_model ):
            saver = tf.train.Saver()
            # Step-1 load graph
            saver = tf.train.import_meta_graph(self.modelPath+self.modelName+'.meta')
            # Step-2: Now let's load the weights saved using the restore method.
            saver.restore(self.session, tf.train.latest_checkpoint(self.modelPath))
            # Accessing the default graph which we have restored
            graph = tf.get_default_graph()
            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            self.y_pred = graph.get_tensor_by_name("y_pred:0")
            ## Let's feed the images to the input placeholders
            self.x= graph.get_tensor_by_name("x:0") 
            self.y_true = graph.get_tensor_by_name("y_true:0")
        
        feed_dict_testing = {self.x: data_in, self.y_true: np.zeros((1, self.nClasses))}
        result = self.session.run(self.y_pred, feed_dict=feed_dict_testing)
        return result
    
    def translateEvaluationResult(self, result):
        
        result = np.array(result)
        out = []
        for elem in result:
            max = -1
            index = -1
            for c in range(0, self.nClasses):
                if(elem[c]>max):
                    max = elem[c]
                    index = c
            out.append(self.classes[index]["name"])
        return out
