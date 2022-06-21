import os, sys
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import random
import dataset

if __name__ == "__main__":
    print(tf.test.is_built_with_cuda())

    data_path = "../MRNet/MRNet-v1.0/train/axial_inv"
    filelist = os.listdir(data_path)
    if '.DS_Store' in filelist:
        filelist.remove('.DS_Store')

    max_slice = 64
    slice_x = 256
    slice_y = 256

    train_generator = dataset.pn_data_generator(data_path, max_slice, slice_x, slice_y, filelist, 1)

    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    vol_shape = (max_slice, slice_x, slice_y)
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

