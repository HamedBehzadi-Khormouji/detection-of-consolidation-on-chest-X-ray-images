'''
Created on Jun 2, 2018

@author: ubuntu
'''
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import  image
import cv2
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input, decode_predictions

class heatmap(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        
        
        
    def load_image_by_path(self, path, imagesize):
        img = image.load_img(path=path, grayscale=True, target_size=imagesize)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x

    def normalize(self,x):
        return x / (k.sqrt(k.mean(k.square(x))) + k.epsilon())

    def create_heat_map2(self, model, layer_name, my_image, img_high, img_width, channel, load_data):

        input_img = model.input
        conv_layer = model.get_layer(layer_name)

        kept_filters = []
        for filter_index in range(215):
            print('Processing filter %d' % filter_index)
            layer_output = conv_layer.output
            loss = k.mean(layer_output[:, :, :, filter_index])
            grads = k.gradients(loss, input_img)[0]
            grads = self.normalize(grads)

            iterate = k.function([input_img], [loss, grads])

            step = 1

            input_img_data=my_image

            for i in range(2):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    break
            if loss_value > 0:
                kept_filters.append((input_img_data, loss_value))

        n = 8

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        img_width=100
        img_height=100
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                width_margin = (img_width + margin) * i
                height_margin = (img_height + margin) * j
                stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

        for i in range(215):
            plt.imshow(stitched_filters[:,:,i])
            plt.show()


    def create_heat_map(self, model, layer_name, my_image, img_high, img_width, channel, load_data):
        x=my_image
        predictions = model.predict_generator(load_data.test_generator)
        filenames = load_data.test_generator.filenames
        print(filenames)
        r = np.argmax(predictions, axis=-1)  # multiple categories
        print('label: ',r)

        img_output = model.output[:, r]
        conv_layer = model.get_layer(layer_name)
        #conv_layer = model.layers[1].get_layer(layer_name)

        grads = k.gradients(img_output, conv_layer.output)[0]
        print('grads shape:',grads.shape)

        pooled_grads = k.mean(grads, axis=(0,1,2))
        print('mean gradiant: ',pooled_grads.shape)

        iterate = k.function([model.input], 
                             [pooled_grads,conv_layer.output[0]])
        
        pooled_grades_value, conv_layer_output_value = iterate([x])


        '''
        
        for i in range(len(conv_layer_output_value)):
            print(pooled_grades_value)
            print(conv_layer_output_value[:,:,i])
            print(conv_layer_output_value[:, :, i] * pooled_grades_value[i])
            print('******************')
        '''



        #for i in range(215):
         #   print(i)
          #  conv_layer_output_value[:, :, i] *= pooled_grades_value[i]
        #print(conv_layer_output_value.shape)


        heatmapres = np.mean(conv_layer_output_value, axis = -1,)
        print(np.sum(heatmapres))
        heatmapres = np.maximum(heatmapres, 0)
        heatmapres /= np.max(heatmapres)

        print(heatmapres)
        
        return heatmapres
    
    
    def super_impose_heatmap(self, img_path, heatmapres):
        img = cv2.imread(img_path)
        plt.imshow(img)
        plt.show()
        heatmap = cv2.resize(heatmapres, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)#cv2.COLORMAP_JET
        heatmap = cv2.addWeighted(heatmap, 0.3, img, 0.5, 0)

        '''
        
        #img = cv2.resize(img,(100,100))
        heatmapres = cv2.resize(heatmapres, (img.shape[1], img.shape[0]))
        heatmapres = np.uint8(255 * heatmapres)
        plt.imshow(heatmapres)
        plt.show()
        heatmapres = cv2.applyColorMap(heatmapres, cv2.COLORMAP_JET)
        plt.imshow(heatmapres)
        plt.show()
        super_imposed_image = cv2.addWeighted(heatmapres, 0.3, img, 0.5, 0)## best paramtere respectively 0.3 0.5
        #super_imposed_image =    heatmapres * 0.4 + img
        '''
        
        return heatmap
    