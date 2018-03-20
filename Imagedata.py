import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import json
import fnmatch

class ImageDataset():

    #CONSTANTS

    #ATTRIBUTES
    # self.classes                  -> classes in config.json
    # self.num_classes              -> number of classes in config.json
    # self.img_path_of_class        -> path of images of that class
    # self.imgFormat                -> format .jpg of images
    # self.imgSize                  -> image to resize the image
    # self.num_img_evalute          -> number of images to preserve to evaluation
    # self.imgNumChannels           -> number of images' channels 

    ###GET the classes to computize (list of strings)
    def __init__(self):
        self.__getClassesConfig__()
        self.___getFormatImg__()
        self.__getImgSize__()
        self.__getImgNumChannels__()


    ### read the configuration from the configuration file and returns the path of every class data
    def __getClassesConfig__(self) :

        data = json.load(open(".\\config.json"))
        self.classes = []
        self.num_classes = 0
        self.img_path_of_class = [] 
        self.num_img_evalute = []
        for classe in data["classes"]:
            self.classes.append(classe["name"])
            self.img_path_of_class.append(classe["img_path"])
            self.num_img_evalute.append(classe["num_valuate"])
        self.num_classes = len(self.classes)

        return self.classes, self.num_classes, self.img_path_of_class
    
    ### reads the image format
    def ___getFormatImg__(self):

        data = json.load(open(".\\config.json"))
        self.imgFormat = data["img_format"]

        return self.imgFormat

    ### reads the imagesize
    def __getImgSize__(self):

        data = json.load(open(".\\config.json"))
        self.imgSize = data["img_size"]

    ### reads the image channels
    def __getImgNumChannels__(self):

        data = json.load(open(".\\config.json"))
        self.imgNumChannels = data["img_num_channels"]

    ### get a list of images from the path
    def getListofFiles(self, path):

        format = ".jpg"
        if (hasattr(self, "imgFormat")):
            format = self.imgFormat
        
        return fnmatch.filter(os.listdir(path), '*'+format)

    ### load an image from path and resizes it 
    def getResizedImage(self, path, img_size):

        image = cv2.imread(path)
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0/255.0) 
        return image

    ### get images from the classname, max_num is the max number of images to load, minimum is minimum index to catch
    def getImagesofClass(self, classname, minimum = 0,  max_num = -1):
        
        index = self.classes.index(classname)
        filelist = self.getListofFiles(self.img_path_of_class[index])
        images = []
        #max num is the maximum number of images to load
        if(max_num != -1 and max_num < len(filelist)):
            filelist = filelist[0:max_num]
        if(minimum > 0 and minimum < len(filelist) ):
            filelist = filelist[ minimum:]
        for name in filelist:
            image = self.getResizedImage( self.img_path_of_class[index] +"\\"+ name, self.imgSize )
            images.append(image)
        
        return images


    ### creates size labels of the classname
    def createLabels(self, size, classname):
        
        index = self.classes.index(classname)
        labels = []
        for a in range(0, size):
            label = np.zeros(self.num_classes)
            label[index] = 1.0
            labels.append(label)

        return labels

    ### returns training images and labels
    def getTrainingData(self, mischia = True):
        
        images = []
        labels = []
        for clas in self.classes:
            partial = self.getImagesofClass(clas, self.num_img_evalute[self.classes.index(clas)])
            labels = labels + self.createLabels(len(partial), clas)
            images = images + partial

        if(mischia == True): #I know it's Horrible
            images, labels = shuffle(images, labels)

        return images, labels
    
    ### returns Valuation images and labels
    def getValuationData(self, mischia = True):
        
        images = []
        labels = []
        for clas in self.classes:
            partial = self.getImagesofClass(clas, 0, self.num_img_evalute[self.classes.index(clas)])
            labels = labels + self.createLabels(len(partial), clas)
            images = images + partial

        if(mischia == True): #I know it's Horrible
            images, labels = shuffle(images, labels)

        return images, labels
    
    ### load an image for evaluation
    def loadEvaluation(self, path):
        imagesq = []
        # Reading the image using OpenCV
        image = cv2.imread(path)
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (self.imgSize, self.imgSize),0,0, cv2.INTER_LINEAR)
        imagesq.append(image)
        imagesq = np.array(imagesq, dtype=np.uint8)
        imagesq = imagesq.astype('float32')
        imagesq = np.multiply(imagesq, 1.0/255.0) 
        #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        return imagesq.reshape(1, self.imgSize, self.imgSize, self.imgNumChannels)

    


