from CNNetwork import CNNetwork
import sys


DATASET = 'skin_binary_class'
TXT_DESCRIPTION = 'dropout 0.5'
IMG_HIGH = 224
IMG_WIDTH = 224
LEARNING_RATE = 1e-3
BATCH_SIZE = 50
STEP_PER_EPOCH = int(7017/BATCH_SIZE)+1 if 7017 % BATCH_SIZE != 0 else int(7017/BATCH_SIZE)
VALIDATION_STEP = int(2000/BATCH_SIZE)+1 if 2000 % BATCH_SIZE != 0 else int(2000/BATCH_SIZE)
TEST_step = int(998/BATCH_SIZE)+1 if 998 % BATCH_SIZE != 0 else int(998/BATCH_SIZE)
CLASSES = 3
EPOCHS = 30

RUNNING_NET = ['ResNet50', 'DenseNet121', 'DenseNet201', 'InceptionV3', 'VGG16', 'VGG19', 'Xception', 'MobileNet', 'InceptionResNetV2']


base_model = 'VGG16'


cnn = CNNetwork(img_high=IMG_HIGH, img_width=IMG_WIDTH, batch_size=BATCH_SIZE, dataset=DATASET,
                classes=CLASSES, description=TXT_DESCRIPTION)
cnn.LoadTrainDataGenerator()
cnn.LoadValidationDataGenerator()
cnn.LoadTestDataGenerator()

cnn.EmptyTempFolder()
cnn.InitModel(base_model=base_model)
cnn.FitModel(epochs=EPOCHS, step_per_epoch=STEP_PER_EPOCH, validation_step=VALIDATION_STEP, learning_rate=LEARNING_RATE)
#cnn.LoadModel()
#cnn.EvaluateModel()
cnn.SaveFigure()
cnn.SaveResults()
cnn.roc_confusion_matrix(TEST_step)


