from keras.models import load_model
from keras import callbacks,optimizers,models,layers
import  matplotlib.pyplot as plt
import statistics
import os
from keras import regularizers
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np
from keras import backend as k


class re_train():

    def __init__(self,CLASSES,epochs, step_per_epoch, validation_step, learning_rate,directories,load_data,trainable_layers,mask_size):
        self.classes = CLASSES
        self.directories = directories
        self.load_data = load_data
        self.epochs = epochs
        self.step_per_epoch = step_per_epoch
        self.validation_step = validation_step
        self.learning_rate = learning_rate
        self.trainable_layers = trainable_layers
        self.net_name=''
        self.mask_size=mask_size

    def global_average_pooling(self,x):
        return k.mean(x, axis=(2, 3))

    def global_average_pooling_shape(self,input_shape):
        return input_shape[0:2]

    def load_model(self, net_name):

        self.model = load_model(os.path.join(self.directories.model_dir, net_name + '.h5'))

        self.model.summary()
        self.model.trainable = True
        new_model = models.Model(self.model.input, self.model.layers[-2].output)

        x = new_model.get_layer('dense_1').output

        #x = layers.Dense(1800,activation='relu')(x)
        #x = layers.Dense(1024,activation='relu')(x)
        x = layers.Dense(512, activation='relu',name='dense_2')(x)
        x = layers.Dense(256, activation='relu',name='dense_3')(x)
        #x = layers.Dense(128, activation='relu')(x)
        '''
        x = layers.Conv2D(216, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                      bias_initializer='zeros',name='conv2d_14')(x)

        x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                     beta_initializer='zeros', gamma_initializer='ones',
                                                     moving_mean_initializer='zeros', moving_variance_initializer='ones'
                                      ,name='batch_normalization_14')(x)
        x = layers.Conv2D(216, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                          bias_initializer='zeros',name='conv2d_15')(x)
        x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                      beta_initializer='zeros', gamma_initializer='ones',
                                      moving_mean_initializer='zeros', moving_variance_initializer='ones'
                                      ,name='batch_normalization_15')(x)
        x = layers.Conv2D(216, self.mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                          bias_initializer='zeros',name='conv2d_16')(x)
        x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                      beta_initializer='zeros', gamma_initializer='ones',
                                      moving_mean_initializer='zeros', moving_variance_initializer='ones'
                                      ,name='batch_normalization_16')(x)
        x = layers.MaxPool2D(pool_size=(2,2),name='mp2')(x)
        x = layers.Flatten()(x)
        #x = layers.Dense(600, activation='relu')(x)
        '''
        output = layers.Dense(self.classes, activation='softmax',name='final_dense')(x)

        new_model = models.Model(inputs=new_model.input, outputs=output)
        self.model = new_model

        self.model.summary()

    def fit_model(self):

        print()
        ''' 
        for layer in self.model.layers:
            layer.trainable = True
            print(layer.name)
            print(layer.trainable)
            if layer.name == 'densenet121':
               print('*********')
               print('*********')
               for l in layer.layers:
                   print(l.name)
                   print(l.trainable)
               print('*********')
               print('*********')
        '''



        print('- Trainable Weights after freezing: {0}'.format(len(self.model.trainable_weights)))

        optmz = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5, amsgrad=False)
        self.model.compile(optimizer=optmz,
                           loss='categorical_crossentropy',
                           # loss='binary_crossentropy',
                           metrics=['accuracy'])
        path = os.path.join(self.directories.re_trained_model_dir  + 'weights-improvement-{epoch:02d}-{val_acc:.2f}' + '.h5')
        print(path)
        checkpoint = callbacks.ModelCheckpoint(path, monitor='accuracy', verbose=0, save_best_only=False,
                                               save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]
        #class_weight = {0: 0.5, 1: 2}

        history = self.model.fit_generator(self.load_data.train_generator,
                                           epochs=self.epochs,
                                           steps_per_epoch=self.step_per_epoch,
                                           validation_data=self.load_data.validation_generator,
                                           validation_steps=self.validation_step,
                                           callbacks=callbacks_list,
                                           #class_weight=class_weight
                                           )

        self.model.save(os.path.join(self.directories.re_trained_model_dir, self.net_name + '.h5'))

        self.acc = history.history['acc']
        self.val_acc = history.history['val_acc']
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']



    def save_figure(self, mean_lenght=20):
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


        plt.title('Training and validation accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid()
        plt.savefig(os.path.join(self.directories.result_dir, self.directories.filename_seq + '-accuracy.png'))

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


        plt.title('Training and Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.grid()
        plt.savefig(os.path.join(self.directories.result_dir, self.directories.filename_seq + '-loss.png'))
        plt.show()


    def roc_confusion_matrix(self,net_name):

        print(self.load_data.validation_dir)
        predict = self.model.predict_generator(self.load_data.test_generator)
        fpr1, tpr1, thersholds1 = roc_curve(self.load_data.test_generator.classes, predict[:, 0])
        fpr2, tpr2, thersholds2 = roc_curve(self.load_data.test_generator.classes, predict[:, 1])

        max_indx = np.argmax(predict, axis=1)
        tn, fp, fn, tp = confusion_matrix(self.load_data.test_generator.classes, max_indx).ravel()

        print('tn = ', tn)
        print('fp = ', fp)
        print('fn = ', fn)
        print('tp = ', tp)

        print('tnr = ', tn / (tn + fp))
        print('fpr = ', fp / (fp + tn))
        print('fnr = ', fn / (fn + tp))
        print('tpr = ', tp / (tp + fn))
        print('acc = ', (tp + tn) / (tp + tn + fp + fn))


        
        self.auc_score = auc(fpr1, tpr1)
        print('AUC = {0}'.format(round(self.auc_score, 4)))
        plt.figure(2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr1, tpr1, 'r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1)
        plt.title('ROC curve (AUC = {0})'.format(round(self.auc_score, 4)))
        plt.grid()
        plt.savefig(os.path.join(self.directories.result_dir, net_name + '-roc.png'))


        self.auc_score = auc(fpr2, tpr2)
        print('AUC = {0}'.format(round(self.auc_score, 4)))

        
        plt.figure(3)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr2, tpr2, 'r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0,1)
        plt.title('ROC curve (AUC = {0})'.format(round(self.auc_score, 4)))
        plt.grid()
        plt.savefig(os.path.join(self.directories.result_dir, net_name + '-roc.png'))

        plt.show()
