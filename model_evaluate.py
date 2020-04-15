from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from heatmap import heatmap
from keras.preprocessing import image
from keras_applications.vgg16 import preprocess_input
#from keras.utils import plot_model
import os
from keras import backend as k
import cv2
import shutil

class model_evaluate():

    def __init__(self,directories,data):
        self.model = []
        self.directories = directories
        self.data = data



    def loadModel(self, net_name):
        print(os.path.join(self.directories.model_dir, net_name + '.h5'))
        self.model = load_model(os.path.join(self.directories.model_dir, net_name + '.h5'))
        self.model.summary()
        #plot_model(self.model, 'model.pdf', show_shapes=True)


    def roc_confusion_matrix(self,net_name,directories):


        predict = self.model.predict_generator(self.data.test_generator)

        filenames = self.data.test_generator.filenames


        fpr2, tpr2, thersholds2 = roc_curve(self.data.test_generator.classes, predict[:, 1])

        max_indx = np.argmax(predict, axis=1)


        

        for i in range(len(filenames)):
            print('name of image: ',filenames[i])
            print('prdicted class: ',max_indx[i])
            print('*******************')
            img_name = filenames[i].split('/')[1]
            img_class = filenames[i].split('/')[0]
            pred_ind = max_indx[i]

            '''
            
            if int(img_class)== 0 and pred_ind ==1:
                img_dir = '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/big_Normal/'+ img_name
                preproc_img_dir = directories.dataset_dir+'/test_heatmap/0/' + img_name
                shutil.copy2(img_dir,'/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/FP/')
                shutil.copy2(preproc_img_dir,'/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/FP_CLAHEHM/')
            else:
                if int(img_class)==0 and pred_ind==0:
                    img_dir = '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/big_Normal/' + img_name
                    preproc_img_dir = directories.dataset_dir + '/test_heatmap/0/' + img_name
                    shutil.copy2(img_dir,
                                 '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/TN/')
                    shutil.copy2(preproc_img_dir,'/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/TN_CLAHEHM/')
                else:
                    if int(img_class)==1 and pred_ind==0:
                        img_dir = '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/classes/1/' + img_name
                        shutil.copy2(img_dir,
                                 '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/FN/')
                    else:
                        if int(img_class)==1 and pred_ind==1:
                            img_dir = '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/classes/1/' + img_name
                            shutil.copy2(img_dir,
                                 '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/TP/')
                            preproc_img_dir = directories.dataset_dir + '/test/1/' + img_name
                            shutil.copy2(preproc_img_dir,'/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/TP_CLAHEHM/')
            '''


        tn, fp, fn, tp = confusion_matrix(self.data.test_generator.classes, max_indx).ravel()

        print('tn = ', tn)
        print('fp = ', fp)
        print('fn = ', fn)
        print('tp = ', tp)

        print('tnr = ', tn / (tn + fp))
        print('fpr = ', fp / (fp + tn))
        print('fnr = ', fn / (fn + tp))
        print('tpr = ', tp / (tp + fn))
        print('acc = ', (tp + tn) / (tp + tn + fp + fn))


        
        #self.auc_score = auc(fpr1, tpr1)
        #print('AUC = {0}'.format(round(self.auc_score, 4)))
        #plt.figure(2)
        #plt.plot([0, 1], [0, 1], 'k--')
        #plt.plot(fpr1, tpr1, 'r')
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.ylim(0, 1)
        #plt.title('ROC curve (AUC = {0})'.format(round(self.auc_score, 4)))
        #plt.grid()
        #plt.savefig(os.path.join(self.directories.result_dir, net_name + '-roc.png'))


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


    def visualize_class_activation_map(self, img_path,IMG_HIGH,IMG_WIDTH):
        model = self.model#load_model(model_path)
        original_img = cv2.imread(img_path, 1)
        base_img = original_img
        original_img = cv2.resize(original_img, (IMG_WIDTH,IMG_HIGH))
        base_img = original_img
        original_img = image.img_to_array(original_img)
        original_img = np.expand_dims(original_img, axis=0)
        #original_img = preprocess_input(original_img)
        img=original_img
        print(img.shape)



        class_weights = model.layers[-2].get_weights()[0]


        final_conv_layer = model.get_layer("conv2d_15")#get_output_layer(model, "conv2d_9")


        get_output = k.function([model.layers[0].input],
                                [final_conv_layer.output,
                                 model.layers[-1].output])


        [conv_outputs, predictions] = get_output([img])


        conv_outputs = conv_outputs[0, :, :, :]


        # Create the class activation map.
        print('conv_outputs.shape ',conv_outputs.shape)
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        target_class = np.argmax(predictions, axis=-1)  # multiple categories
        print(target_class)

        for i, w in enumerate(class_weights[:, target_class]):
            cam += w * conv_outputs[:, :, i]

        cam /= np.max(cam)
        w,h,j = base_img.shape


        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        #print(heatmap)
        heatmap[np.where(cam < 0.02)] = 0

        #base_img = heatmap * 0.5  + base_img
        #img = img[0,:,:,:]
        heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HIGH))

        print(heatmap.shape)
        print(base_img.shape)

        img = cv2.addWeighted(heatmap, 0.5, base_img, 0.9, 0)

        #plt.imshow(base_img, alpha=0.9)
        #plt.imshow(heatmap, cmap='jet', alpha=0.5)

        #plt.imshow(heatmap)
        #plt.show()
        plt.imshow(img)
        plt.show()

        '''
        print(cam)
        hmap = heatmap()
        img_heatmap = hmap.super_impose_heatmap(img_path, cam)

        plt.imshow(cam)
        plt.show()
        plt.imshow(img_heatmap)
        plt.show()
        '''

    def G_CAM(self,img_path,IMG_HIGH,IMG_WIDTH,layer_name):
        IMG = image.load_img(img_path, target_size=(IMG_HIGH, IMG_WIDTH))
        plt.show()
        img = image.img_to_array(IMG)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        predictions = self.model.predict_generator(self.data.test_generator)
        filenames = self.data.test_generator.filenames
        print(filenames)
        r = np.argmax(predictions, axis=-1)  # multiple categories
        print('label: ', r)

        img_output = self.model.output[:, r]
        conv_layer = self.model.get_layer(layer_name)
        # conv_layer = model.layers[1].get_layer(layer_name)

        grads = k.gradients(img_output, conv_layer.output)[0]
        print('grads shape:', grads.shape)

        pooled_grads = k.mean(grads, axis=(0, 1, 2))
        print('mean gradiant: ', pooled_grads.shape)

        iterate = k.function([self.model.input],
                             [pooled_grads, conv_layer.output[0]])

        pooled_grades_value, conv_layer_output_value = iterate([img])

        print(pooled_grades_value.shape)
        print(conv_layer_output_value.shape)

        output= conv_layer_output_value[0, :]

        weights = pooled_grades_value#np.mean(grads_val, axis=(0, 1))
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam /= np.max(cam)
        w, h, j = IMG.shape
        cam = cv2.resize(cam, (h, w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_RAINBOW)
        # print(heatmap)
        heatmap[np.where(cam < 0.02)] = 0

        # img = heatmap * 0.5  + img
        img = img[0, :, :, :]

        print(heatmap.shape)
        print(IMG.shape)

        img = cv2.addWeighted(heatmap, 0.5, IMG, 0.9, 0)

        plt.imshow(heatmap)
        plt.show()
        plt.imshow(img)
        plt.show()


    def display_heatmap2(self,img_path,img_high,img_width,layer_name,load_data):

        IMG = image.load_img(img_path, target_size=(img_high, img_width))
        plt.show()
        img = image.img_to_array(IMG)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        hmap = heatmap()
        img_heatmap = hmap.create_heat_map2(self.model, layer_name, img, img_high, img_width, 3, load_data)
        print('heatmap')
        plt.imshow(img_heatmap)
        plt.show()
        img_heatmap = hmap.super_impose_heatmap(img_path, img_heatmap)

        plt.imshow(img_heatmap)
        plt.show()

    def display_heatmap(self,img_path,img_high,img_width,layer_name,load_data):

        IMG = image.load_img(img_path, target_size=(img_high, img_width))
        plt.show()
        img = image.img_to_array(IMG)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        hmap = heatmap()
        img_heatmap = hmap.create_heat_map(self.model, layer_name, img, img_high, img_width, 3, load_data)
        print('heatmap')
        plt.imshow(img_heatmap)
        plt.show()
        img_heatmap = hmap.super_impose_heatmap(img_path, img_heatmap)

        plt.imshow(img_heatmap)
        plt.show()

    def evaluate_model(self):

        self.eval_score, self.eval_acc = self.model.evaluate_generator(self.data.test_generator)

        print('Evaluate Score = {0}'.format(self.eval_score))
        print('Evaluate Accuracy = {0}'.format(self.eval_acc))

    def comparison_models(self,nets_name,input_size,labels):

        models = []
        colors = ['r','y','g','b','k','m']
        indx=0
        plt.figure(3)
        ax = plt.gca()
        plots = []
        f = open('/home/atlas/PycharmProjects/SimpleNet/all.txt', 'w')
        for net_name in nets_name:
            print(net_name)
            if 'Extra_Validation_On_ChestNet' in net_name :
                models.append(load_model(os.path.join(self.directories.model_dir, 'chest' + '.h5')))
            else:
                models.append(load_model(os.path.join(self.directories.model_dir, net_name + '.h5')))

        cnt=1;
        for iii in range(len(models)):
            model = models[iii]
            net_name = nets_name[iii]
            print(net_name)
            self.data.img_high = input_size[indx]
            self.data.img_width = input_size[indx]
            if net_name == 'Extra_Validation_On_ChestNet':
                self.data.LoadTestDataGenerator_extra_val()
            else:
                self.data.LoadTestDataGenerator()
            predict = model.predict_generator(self.data.test_generator)
            fpr, tpr, thersholds2 = roc_curve(self.data.test_generator.classes, predict[:, 1])
            model=None
            self.auc_score = auc(fpr, tpr)
            print(self.auc_score)
            for i in range(len(fpr)):
                f.write(str(fpr[i]))
                f.write(',')
            f.write('\n')
            f.write('******')
            f.write('\n')
            for i in range(len(tpr)):
                f.write(str(tpr[i]))
                f.write(',')
            f.write('\n')
            f.write('!!!!!!!!!!!!!!!!!')
            f.write('\n')
            '''
            
            plt.plot([0, 1], [0, 1], 'k--')
            curv = ax.plot(fpr, tpr, colors[indx],label=labels[indx])
            plots.append(curv)
            plt.xlabel('False Positive Rate (Fall-out)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.ylim(0, 1)
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.grid()
            indx = indx + 1
            cnt = cnt+1
            '''
        f.close()
        '''
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.directories.result_dir, 'ROC_Models' + '.png'))
        '''

    def plotROC(self,labels):


        fpr=[]
        tpr=[]
        colors = ['g', 'k', 'r', 'b', 'y', 'm',[0,0.5,0.5],[0.7,0.3,0.1]]
        #colors =[[1,0,0],[0,1,0],[0,0,1],
         #        [0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]
         #        ,[0.7,0.3,0.1],[0.3,0.7,0.1]]
        real_label=[0,0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0]
        rdg_1 =    [0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,1,0,0,1,0]
        rdg_2 =    [0,0,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1]
        indx = 0
        plt.figure(3)
        ax = plt.gca()
        plots = []

        text_file = open('/home/atlas/PycharmProjects/SimpleNet/VGG_ROC.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))

        text_file = open('/home/atlas/PycharmProjects/SimpleNet/chest2.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))

        text_file = open('/home/atlas/PycharmProjects/SimpleNet/chest.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))

        text_file = open('/home/atlas/PycharmProjects/SimpleNet/BreastChest.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))


        text_file = open('/home/atlas/PycharmProjects/SimpleNet/DenseNet.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))


        text_file = open('/home/atlas/PycharmProjects/SimpleNet/chest_extra_val.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        for i in range(len(lines)):
            if i % 2 == 0:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                fpr.append(tempist)
            else:
                temp = lines[i].split(',')
                tempist = []
                for j in range(len(temp)):
                    if temp[j] != '\n':
                        tempist.append(float(temp[j]))
                tpr.append(tempist)

        print(len(fpr))
        print(len(tpr))

        rdg_fpr, rdg_tpr, thersholds1 = roc_curve(real_label, rdg_1)
        auc_score = auc(rdg_fpr, rdg_tpr)
        print(auc_score)
        tpr.append(rdg_tpr)
        fpr.append(rdg_fpr)

        rdg_fpr, rdg_tpr, thersholds1 = roc_curve(real_label, rdg_2)
        auc_score = auc(rdg_fpr, rdg_tpr)
        print(auc_score)
        tpr.append(rdg_tpr)
        fpr.append(rdg_fpr)

        markers=['+','s','h','*','v','<','.','o']

        for i in range(len(fpr)):
            plt.plot([0, 1], [0, 1], 'k--')
            curv = ax.plot(fpr[i], tpr[i], color=colors[indx], label=labels[indx],marker=markers[indx])
            plots.append(curv)
            plt.xlabel('False Positive Rate (Fall-out)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.ylim(0, 1)
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.grid()
            indx=indx+1

        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.directories.result_dir, '_Models_ROC_' + '.png'))