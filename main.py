from simpNet import simpNet
from make_directories import make_directories
from model_evaluate import model_evaluate
from make_load_date import make_load_date
from re_train import re_train



'''
Parameters of Network

'''

dropout = 'dropout 0.5'# this is an discription that shows your model uses dropout layer or not.
IMG_HIGH = 224# the hight of your image
IMG_WIDTH = 224#the width of your iimage
LEARNING_RATE = 1e-3# learning rate for your optimization method
BATCH_SIZE = 20
STEP_PER_EPOCH =  250
VALIDATION_STEP = 30
CLASSES = 2
EPOCHS =30
mask_size = 7# the mask size of your convolutional layer

'''
Parameters of program 
'''
DATASET = 'CLAHEHM_Normal_Pneumonia'# the name of folder including your datset categorized as train, validation and test

'''
this attribute control your mode of code. it can take 3 valuses train, when you use the code for train, re_train, when you use the pretrained model for a new train or evaluate, whe you want to evaluate the trained model on test data anf get ROC figure
'''
Mode = 'evaluate'  

'''
Parameters of Evaluate Mode
'''
Net = 'VGG16'#weights-improvement-50-0.92


'''
Parameters of Re_train Mode
this is used for re-train mode. the trainable_layers attribute contains the name of layers that you want tu turn on their trainable mode
'''
trainable_layers = ['conv2d_7','conv2d_8','conv2d_9','conv2d_10','conv2d_11','conv2d_12','conv2d_13','dense_1','dense_2']




if Mode == 'train':

    sim_net = simpNet(mask_size = mask_size, img_high=IMG_HIGH, img_width=IMG_WIDTH, batch_size=BATCH_SIZE, dataset=DATASET,
                    classes=CLASSES, description=TXT_DESCRIPTION)
	
    sim_net.LoadTrainDataGenerator()#load train data and apply the augmentation
    sim_net.LoadValidationDataGenerator()#load validation data and apply the augmentation
    sim_net.LoadTestDataGenerator()##load test data and apply the augmentation
    sim_net.EmptyTempFolder()
    sim_net.InitModel()#in this method you can design your network
    sim_net.FitModel(epochs=EPOCHS, step_per_epoch=STEP_PER_EPOCH, validation_step=VALIDATION_STEP, learning_rate=LEARNING_RATE)# this function starts training process
    sim_net.SaveFigure()

if Mode == 're_train':
    directories = make_directories(dataset=DATASET)#make an object of your directory class
    load_data = make_load_date(IMG_HIGH, IMG_WIDTH, BATCH_SIZE, directories)# this method load your dataset
    rt = re_train(CLASSES,EPOCHS, STEP_PER_EPOCH, VALIDATION_STEP, LEARNING_RATE,directories,load_data,trainable_layers,mask_size)
    rt.load_model(Net)#this method load your saved medol .h5
    rt.fit_model()$this function starts training process
    rt.save_figure()
    rt.roc_confusion_matrix(Net)# this function compute the confusion matrix of your model

if Mode == 'evaluate':
    directories = make_directories(dataset=DATASET)
    load_data = make_load_date(IMG_HIGH, IMG_WIDTH, BATCH_SIZE, directories)
    evl = model_evaluate(directories,load_data)#this method evaluate your model using test data
    evl.plotROC()#this method plot ROC figure


