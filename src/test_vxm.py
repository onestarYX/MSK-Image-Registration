import os, sys
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
import random
from scipy.ndimage.interpolation import zoom
from train_vxm_mrnet import vxm_data_generator

if __name__ == "__main__":
    print("Using CUDA: ", tf.test.is_built_with_cuda())

    data_path = "../MRNet/MRNet-v1.0/train/axial"
    filelist = os.listdir(data_path)
    filelist.remove('.DS_Store')

    max_slice = 128
    x_size = 256
    y_size = 256

    valid_generator = vxm_data_generator(data_path, max_slice, x_size, y_size, filelist, 1)
    val_input, val_output = next(valid_generator)

    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    vol_shape = (max_slice, x_size, y_size)
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    checkpoint_path = "../model_cp/vxm_upscaling.ckpt"
    vxm_model.load_weights(checkpoint_path)

    val_pred = vxm_model.predict(val_input)

    # Visualization
    images = [img[0, max_slice // 2, :, :, 0] for img in val_input + val_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)