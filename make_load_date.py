from keras.preprocessing import image
import keras

class make_load_date():
    def __init__(self,img_high, img_width, batch_size,directories):
        self.img_high = img_high
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_dir = directories.train_dir
        self.validation_dir = directories.validation_dir
        self.test_dir = directories.test_dir
        self.train_generator = []
        self.validation_generator = []
        self.test_generator = []

        self.LoadTrainDataGenerator()
        self.LoadValidationDataGenerator()
        self.LoadTestDataGenerator()



    def LoadTrainDataGenerator(self):
        train_datagen = image.ImageDataGenerator(
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
        )



    def LoadValidationDataGenerator(self):
        validation_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
        )
        #self.validation_generator = self.GenerateImage(validation_datagen)
        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            #save_to_dir='temp',
            #save_format='png'
            # color_mode='grayscale'
        )

    def LoadTestDataGenerator(self):
        test_datagen = image.ImageDataGenerator(
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
    def LoadTestDataGenerator_extra_val(self):
        test_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
        )
        self.test_generator = test_datagen.flow_from_directory(
            '/home/atlas/PycharmProjects/DataSets/chest8_Pneumonia_Normal/cropped/CLAHEHM/test/',
            target_size=(self.img_high, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            # save_to_dir='temp',
            # save_format='png'
        )
