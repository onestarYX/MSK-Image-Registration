import random
import sys

import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import scipy.interpolate as interpolate

# sys.path.append('./ext/neuron')
# sys.path.append('./ext/neuron/neuron')
# sys.path.append('./ext/pynd-lib')
# sys.path.append('./ext/pytools-lib')
# import neuron.neuron.layers as nrn_layers

img1_path = 'data/knee1FS time1 337172.nii.gz'
img2_path = 'data/knee1FS time2 338366.nii.gz'
img3_path = 'data/Knee2NFS Time1 20090305.nii.gz'
img4_path = 'data/Knee2NFS Time2 20140523.nii.gz'

img1Obj = nib.load(img1_path)
img2Obj = nib.load(img2_path)
img3Obj = nib.load(img3_path)
img4Obj = nib.load(img4_path)

img1 = img1Obj.get_fdata()
img1 = np.reshape(img1, (320, 320, 192))
img2 = img2Obj.get_fdata()
img2 = np.reshape(img2, (320, 320, 192))
img3 = img3Obj.get_fdata()
img3 = np.reshape(img3, (320, 320, 192))
img4 = img4Obj.get_fdata()
img4 = np.reshape(img4, (320, 320, 192))

movingList = [img1, img3]
moving = np.array(movingList)
print(moving.shape)

fixedList = [img2, img4]
fixed = np.array(fixedList)

vol_size = (320, 320, 192)
# Do we want the src to be (320, 320, 192) or (320, 320, 192, 1)?
src = keras.Input([*vol_size, 1])
tgt = keras.Input([*vol_size, 1])
concat_input = layers.concatenate([src, tgt])
print(concat_input.shape)

conv_layer_1 = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), input_shape=[*vol_size, 2], activation='relu')(concat_input)
conv_layer_2 = layers.Conv3D(filters=1, kernel_size=(5, 5, 5), activation="relu")(conv_layer_1)
conv_layer_3 = layers.Conv3D(filters=1, kernel_size=(20, 20, 20), activation="relu")(conv_layer_2)
dense_layer_1 = layers.Flatten()(conv_layer_3)
dense_layer_2 = layers.Dense(58982400, activation="relu")(dense_layer_1)
# reshaped_output = layers.Reshape((*vol_size, 3))(dense_layer_2)

# random_flow_list = []
# for i in range(vol_size[0] * vol_size[1] * vol_size[2] * 3):
#     random_flow_list.append(random.uniform(20, 25))
#
# random_flow = np.array(random_flow_list)
# random_flow = np.reshape(random_flow, (*vol_size, 3))
# print(random_flow.shape)
#
#
# xx = np.arange(vol_size[1])
# yy = np.arange(vol_size[0])
# zz = np.arange(vol_size[2])
# grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
# print(grid[1,0,2])
#
# sample = random_flow+grid
# sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
# warped = interpolate.interpn((yy, xx, zz), img1, sample, method='nearest', bounds_error=False, fill_value=0)
# print(warped.shape)
# plt.imshow(warped[:,:,100])
# plt.show()

# Write the img array to a txt file (slow and huge, caution!)
# file = open("./data/NetworkInput/sampleKneeInput.txt", "w")
# for img in imgList:
#     for sli in range(0, img.shape[2]):
#         for row in range(0, img.shape[0]):
#             for col in range(0, img.shape[1]):
#                 file.write(str(img[row][col][sli]) + " ")
#             file.write('\n')



