import os

import glob
import shutil

pre_procces_add = '/home/atlas/PycharmProjects/SimpleNet/CLAHEHM_Normal_Pneumonia/test/0/*.jpeg'
original_add_train = '/home/atlas/PycharmProjects/SimpleNet/chest_xray/train/0/'
original_add = '/home/atlas/PycharmProjects/SimpleNet/chest_xray/test/cropped_NORMAL/'
save_add = '/home/atlas/PycharmProjects/SimpleNet/cropped_ChestXRay/test(for_model)/0'

if __name__ == '__main__':

    for file_add in glob.glob(pre_procces_add):
        name = file_add.split('/')[-1]
        img_add = original_add + name
        if os.path.isfile(img_add) :
            shutil.copy(img_add,save_add)
        else:
            print(img_add)

