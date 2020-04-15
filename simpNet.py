from keras.utils import plot_model
import os
import numpy as np
from keras.preprocessing import image
from keras import models,callbacks
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import time
import statistics
import csv
from sklearn.metrics import roc_curve, auc
#from sklearn.ensemble import RandomForestClassifier
import glob
import pathlib
import fnmatch
from keras.models import load_model
import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121


class simpNet:

    def __init__( self, mask_size = 5, img_high=224, img_width=224, batch_size=32, dataset='dataset', classes=2, description=''):
        self.img_high = img_high
        self.img_width = img_width

        self.mask_size = mask_size

        self.description = description
        self.dataset = dataset
        self.classes = classes

        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.dataset_dir = os.path.join('/home/atlas/PycharmProjects/SimpleNet/',self.dataset)
        self.tensorboard_log_dir = os.path.join(self.base_dir, 'log')
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.validation_dir = os.path.join(self.dataset_dir, 'val')
        #self.validation_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        self.temp_dir = os.path.join(self.base_dir, 'temp')
        self.result_dir = os.path.join(self.base_dir, 'results')
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.re_trained_model_dir = os.path.join(self.base_dir, 're_trained_model')

        self.running_time = ''
        #self.running_time_id = ''
        self.filename_seq = self.GetFilenameSequence(self.result_dir)
        self.model = None
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epochs = 0
        self.step_per_epoch = 0
        self.validation_step = 0
        self.learning_rate = 0
        self.val_acc_mean = 0
        self.val_loss_mean = 0
        self.val_acc_min = 0
        self.val_loss_min = 0
        self.val_acc_max = 0
        self.val_loss_max = 0
        self.elapsed_time = 0
        self.batch_size = batch_size
        self.trainable_layers = []
        self.auc_score = 0
        self.eval_score = 0
        self.eval_acc = 0
        self.base_model = 'simNet'
        self.conv_base = None
        self.use_testdata_generator = True
        self.saved_model=[]

        self.running_time = ''
        #self.running_time_id = ''

        #os.environ['Path'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    def GetFilenameSequence(self, directory):
        matches = fnmatch.filter(os.listdir(directory), '*-accuracy.png')
        seq = "{0:04d}".format(len(matches)+1)
        filepath = os.path.join(self.result_dir, seq + '-accuracy.png')
        while os.path.isfile(filepath):
            seq = "{0:04d}".format(int(seq) + 1)
            filepath = os.path.join(self.result_dir, seq + '-accuracy.png')

        return seq
        #return "{0:04d}-accuracy.png".format(len(matches))

    def rgb2gray(self,im):
        if len(im.shape) == 3:
            im = tf.image.rgb_to_grayscale(im)
        return im

    def myScale(self,img):
        max_i = np.max(img)
        return img#img/max_i


    def LoadTrainDataGenerator(self):
        train_datagen = image.ImageDataGenerator( #preprocessing_function = self.myScale,
            rescale=1. / 255,
            #horizontal_flip=True,
            #vertical_flip=True,
        )
        #self.train_generator = self.GenerateImage(train_datagen)
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            #save_to_dir='temp',
            #save_format='png'
            # color_mode='grayscale'

        )

    def LoadValidationDataGenerator(self):
        validation_datagen = image.ImageDataGenerator(#preprocessing_function = self.myScale,
            rescale=1. / 255,
        )
        #self.validation_generator = self.GenerateImage(validation_datagen)
        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,

            #save_to_dir='temp',
            #save_format='png'
            # color_mode='grayscale'
        )

    def LoadTestDataGenerator(self):
        test_datagen = image.ImageDataGenerator(#preprocessing_function = self.myScale,
            rescale=1. / 255,
        )
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,

            #save_to_dir='temp',
            #save_format='png'
        )
        self.use_testdata_generator = True
    '''
    
    def GenerateImage(self, generator):
        genX = generator.flow_from_directory(
            self.train_dir,
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            #save_to_dir='temp',
            #save_format='png'
        )
        while True:
            X, y = genX.next()
            img = X[:, :, 0]
            X[:, :, 1] = img
            X[:, :, 2] = img
            yield (X, y)

    def LoadValidationDataGen(self):
        val_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
        )
        while 1:
            X, y = val_datagen.next()
            img = X[:, :, 0]
            X[:, :, 1] = img
            X[:, :, 2] = img
            yield (X, y)

    def LoadTestData(self):
        # Helper.SampleImage(train_datagen, train_dir+'/calc', (IMG_HIGH, IMG_WIDTH))
        # sampleImage(train_datagen, os.path.join(dataset_dir, 'train', 'mass'))
        # exit()
        self.test_data = self.LoadImageData(self.test_dir,
                                            target_size=(self.img_high, self.img_width)
                                            )
        self.use_testdata_generator = True

    def __array2image(self, arr, out_filename):
        img = Image.fromarray(arr)
        img.save(out_filename)

    def __image2array(self, imgfile, target_size):
        img = Image.open(imgfile)
        img = img.convert('RGB').resize((target_size[1],target_size[0]))
        temp = np.array(img, dtype=np.uint8)
        #result = np.zeros(target_size, dtype=np.uint8)
        #result[:temp.shape[0], :temp.shape[1]] = temp
        return temp

    def LoadImageData(self, directory, target_size=(224, 224)):

        data = {'images': [], 'class_indices': {}, 'classes': []}
        classes = []
        class_indices = {}
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)

        for idx, val in enumerate(classes):
            class_indices[val] = idx

        imagelist = []
        for subdir in classes:
            class_dir = os.path.join(directory, subdir, '**')
            for _pathfile in glob.glob(class_dir, recursive=False):
                try:
                    if os.path.isfile(_pathfile):
                        if pathlib.Path(_pathfile).suffix == '.png' or pathlib.Path(_pathfile).suffix == '.jpg':
                            imagelist.append(self.__image2array(_pathfile, target_size))
                            data['classes'].append(class_indices[subdir])
                except Exception as ex:
                    print(ex)
        temp = np.asarray(imagelist, dtype=np.uint8)
        temp = temp.astype('float32')/255.
        data['images'] = np.reshape(temp, (len(temp), target_size[0], target_size[1], 3))
        data['classes'] = np.asarray(data['classes'], dtype=np.int8)
        #data['images'] = imagelist
        data['class_indices'] = class_indices
        print('Load {0} images belonging to {1} classes.'.format(len(data['images']) , len(data['class_indices'])))

        return data
        
    '''

    def EmptyTempFolder(self):
        for tempfiles in glob.glob(os.path.join(self.temp_dir, '**')):
            os.unlink(tempfiles)

    def InitModel(self):

        self.model = models.Sequential()

        # Conv. 1
        #66
        self.model.add(layers.Conv2D(32, self.mask_size, strides=(1, 1), padding='valid', activation='relu',use_bias=True, bias_initializer='zeros',input_shape=(self.img_high,self.img_width,3)))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        # Conv. 2,3 & 4
        # 64
        self.model.add(layers.Conv2D(64, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(64, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True, bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
        ###############self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(64, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True, bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        # Conv. 5
        #96
        self.model.add(layers.Conv2D(128, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',moving_variance_initializer='ones'))

        # Max Pooling
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))
#        self.model.add(layers.Dropout(0.2))

        
        # Conv. 6,7,8 & 9
        #96
        self.model.add(layers.Conv2D(128, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True, bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(128, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(128, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(128, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.2))


        
        # Conv. 10
        #144
        self.model.add(layers.Conv2D(256, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True, bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))

        
        
        # Max Pooling
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))
#        self.model.add(layers.Dropout(0.3))

        
        # Conv. 11
        #144
        self.model.add(layers.Conv2D(256, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.3))

        
        # Conv. 12
        #178
        self.model.add(layers.Conv2D(256, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True, bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.3))

        
        # Conv. 13
        #216
        self.model.add(layers.Conv2D(512, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#        self.model.add(layers.Dropout(0.3))



        # Conv. 14
        self.model.add(layers.Conv2D(512, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))

        '''
        # Max Pooling
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones'))

        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        
        
        # Conv. 19
        self.model.add(
            layers.Conv2D(512, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                          bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                 beta_initializer='zeros', gamma_initializer='ones',
                                                 moving_mean_initializer='zeros', moving_variance_initializer='ones'))
        # Conv. 20
        self.model.add(
            layers.Conv2D(512, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                          bias_initializer='zeros'))
        self.model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                 beta_initializer='zeros', gamma_initializer='ones',
                                                 moving_mean_initializer='zeros', moving_variance_initializer='ones'))

        '''


        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(1024,activation='relu'))
        #self.model.add(layers.Dense(1800, activation='relu'))
        #self.model.add(layers.Dense(512, activation='relu'))
        #self.model.add(layers.Dense(256, activation='relu'))

        self.model.add(layers.Dense(self.classes, activation='softmax'))

        self.model.summary()






    def FitModel(self, epochs, step_per_epoch, validation_step, learning_rate=1e-5):
        self.running_time = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())

        self.epochs = epochs
        self.step_per_epoch = step_per_epoch
        self.validation_step = validation_step
        self.learning_rate = learning_rate


        #optmz = optimizers.RMSprop(lr=learning_rate)
        optmz = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5, amsgrad=False)
        self.model.compile(optimizer=optmz,
                           loss='categorical_crossentropy',
                           #loss='binary_crossentropy',
                           metrics=['accuracy'])
        path = os.path.join(self.model_dir, self.base_model + self.filename_seq  + 'simple_net-{epoch:02d}-{val_acc:.3f}'+ '.h5')
        checkpoint = callbacks.ModelCheckpoint(path, monitor='accuracy', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]


        #ImageFile.LOAD_TRUNCATED_IMAGES = True

        history = self.model.fit_generator(self.train_generator,
                                           epochs=epochs,
                                           steps_per_epoch=step_per_epoch,
                                           validation_data=self.validation_generator,
                                           validation_steps=validation_step,
                                           callbacks=callbacks_list
                                           )



        self.model.save(os.path.join(self.model_dir, self.base_model + self.filename_seq + '.h5'))
        json_string = self.model.to_json()
        with open(os.path.join(self.model_dir, self.base_model + '.json'), 'w', encoding='utf8') as json_file:
            json_file.write(json_string)

        self.acc = history.history['acc']
        self.val_acc = history.history['val_acc']
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
    def re_train(self):
        pass

    def LoadModel(self,net_name):
        self.saved_model = load_model(os.path.join(self.saved_model_dir, net_name + '.h5'))

    def EvaluateModel(self):
        if self.use_testdata_generator:
            self.eval_score, self.eval_acc = self.model.evaluate_generator(self.test_generator)
        else:
            self.eval_score, self.eval_acc = self.model.evaluate(self.test_data['images'], self.test_data['classes'])
        print('Evaluate Score = {0}'.format(self.eval_score))
        print('Evaluate Accuracy = {0}'.format(self.eval_acc))

    def PredictModel(self):
        if self.use_testdata_generator:
            predict = self.model.predict_generator(self.test_generator)
            fpr1, tpr1, thersholds1 = roc_curve(self.test_generator.classes, predict[:,0])
            fpr2, tpr2, thersholds2 = roc_curve(self.test_generator.classes, predict[:, 1])
        else:
            predict = self.model.predict(self.test_data['images']).ravel()
            fpr, tpr, thersholds = roc_curve(self.test_data['classes'], predict)

        self.auc_score = auc(fpr1, tpr1)
        print('AUC = {0}'.format(round(self.auc_score, 4)))
        plt.figure(2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr1, tpr1, 'r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1)
        plt.suptitle(self.base_model)
        plt.title('ROC curve (AUC = {0})'.format(round(self.auc_score, 4)))
        plt.grid()
        plt.savefig(os.path.join(self.result_dir, self.filename_seq + '-roc.png'))


        self.auc_score = auc(fpr2, tpr2)
        print('AUC = {0}'.format(round(self.auc_score, 4)))
        plt.figure(2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr2, tpr2, 'r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0,1)
        plt.suptitle(self.base_model)
        plt.title('ROC curve (AUC = {0})'.format(round(self.auc_score, 4)))
        plt.grid()
        plt.savefig(os.path.join(self.result_dir, self.filename_seq + '-roc.png'))

        plt.show()
        #pass

    def SaveFigure(self, mean_lenght=20):
        _val_acc_last = self.val_acc[-1 * mean_lenght:len(self.val_acc)]
        _val_loss_last = self.val_loss[-1 * mean_lenght:len(self.val_loss)]
        self.val_acc_mean = round(statistics.mean(_val_acc_last), 3)
        self.val_loss_mean = round(statistics.mean(_val_loss_last), 3)

        self.val_acc_min = round(min(_val_acc_last), 3)
        self.val_loss_min = round(min(_val_loss_last), 3)

        self.val_acc_max = round(max(_val_acc_last), 3)
        self.val_loss_max = round(max(_val_loss_last), 3)

        _epochs = range(1, len(self.acc) + 1)
        plt.figure(1)
        plt.plot(_epochs, self.acc, 'bo', label='Training acc')
        plt.plot(_epochs, self.val_acc, 'b', label='Validation acc')

        plt.plot(_epochs, [self.val_acc_mean] * len(_epochs), ':g')
        plt.text(len(_epochs), self.val_acc_mean, str(self.val_acc_mean),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'green', 'edgecolor': 'green', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.plot(_epochs, [self.val_acc_min] * len(_epochs), ':y')
        plt.text(1, self.val_acc_min, str(self.val_acc_min),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.plot(_epochs, [self.val_acc_max] * len(_epochs), ':c')
        plt.text(len(_epochs) // 2, self.val_acc_max, str(self.val_acc_max),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'cyan', 'edgecolor': 'cyan', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.suptitle(self.base_model)
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.grid()
        plt.savefig(os.path.join(self.result_dir, self.filename_seq + '-accuracy.png'))

        plt.figure(2)
        plt.plot(_epochs, self.loss, 'ro', label='Training loss')
        plt.plot(_epochs, self.val_loss, 'r', label='Validation loss')

        plt.plot(_epochs, [self.val_loss_mean] * len(_epochs), ':g')
        plt.text(len(_epochs), self.val_loss_mean, str(self.val_loss_mean),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'green', 'edgecolor': 'green', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.plot(_epochs, [self.val_loss_min] * len(_epochs), ':y')
        plt.text(1, self.val_loss_min, str(self.val_loss_min),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.plot(_epochs, [self.val_loss_max] * len(_epochs), ':c')
        plt.text(len(_epochs) // 2, self.val_loss_max, str(self.val_loss_max),
                 verticalalignment='center', horizontalalignment='center',
                 bbox={'facecolor': 'cyan', 'edgecolor': 'cyan', 'alpha': 0.5, 'pad': 5},
                 fontsize=10)

        plt.suptitle(self.base_model)
        plt.title('Training and Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.grid()
        plt.savefig(os.path.join(self.result_dir, self.filename_seq + '-loss.png'))
        plt.show()

    def SaveResults(self):
        csvfields = [self.running_time, self.base_model, self.filename_seq,
                     self.batch_size, self.epochs, self.step_per_epoch, self.validation_step,
                     self.val_acc_mean, self.val_loss_mean, self.auc_score,
                     self.eval_score, self.eval_acc,
                     self.learning_rate,
                     '+'.join(self.trainable_layers),
                     len(self.model.trainable_weights),
                     self.img_high, self.img_width,
                     self.elapsed_time,
                     self.dataset, self.description
                     ]
        with open(os.path.join(self.result_dir, 'result.csv'), 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csvfields)

    def BeepAlert(self):
        pass

    def ErrorBeep(self):
       pass


