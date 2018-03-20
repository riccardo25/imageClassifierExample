import Imagedata 
from Model import ModelCNN

#to modify the behaviour of this program, edit the config.json file
#lets acquire photos of classes
print("Loading traing data.... ")

ds = Imagedata.ImageDataset()
train_img, train_labels = ds.getTrainingData()
print("Training data:" + str(len(train_img)))
print("done")
print("Loading evaluation data... ")

evaluate_img, evaluate_labels = ds.getValuationData()

print("done")
print("Creating model...")

cnn = ModelCNN()

print("Let's train!")

cnn.train(500, train_img, train_labels, evaluate_img, evaluate_labels)

print("Train end")
print("Evaluation time:")

image = ds.loadEvaluation(".\\img\\test\\test.jpg")
result = cnn.evaluate(image, True)
print(result)
print(cnn.translateEvaluationResult(result))

