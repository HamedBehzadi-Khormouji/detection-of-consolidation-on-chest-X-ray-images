import os
import fnmatch


class make_directories():

    def __init__(self,dataset='dataset'):

        self.dataset = dataset
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.dataset_dir = os.path.join('/home/atlas/PycharmProjects/SimpleNet/', self.dataset)
        self.tensorboard_log_dir = os.path.join(self.base_dir, 'log')
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.validation_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir =os.path.join(self.dataset_dir, 'test')
        #self.test_dir ='/home/atlas/PycharmProjects/SimpleNet/chest_xray/reults_ChestNet/TP/'#os.path.join(self.dataset_dir, 'test')
        self.temp_dir = os.path.join(self.base_dir, 'temp')
        self.result_dir = os.path.join(self.base_dir, 'results')
        self.filename_seq = self.GetFilenameSequence(self.result_dir)
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.re_trained_model_dir = os.path.join(self.base_dir, 're_trained_model/')

    def GetFilenameSequence(self, directory):
        matches = fnmatch.filter(os.listdir(directory), '*-accuracy.png')
        seq = "{0:04d}".format(len(matches)+1)
        filepath = os.path.join(self.result_dir, seq + '-accuracy.png')
        while os.path.isfile(filepath):
            seq = "{0:04d}".format(int(seq) + 1)
            filepath = os.path.join(self.result_dir, seq + '-accuracy.png')

        return seq

        #return "{0:04d}-accuracy.png".format(len(matches))

