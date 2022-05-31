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

    inv_data_path = "../MRNet/MRNet-v1.0/train/axial_inv"
    filelist_inv = os.listdir(inv_data_path)
    if '.DS_Store' in filelist_inv:
        filelist_inv.remove('.DS_Store')
    start_idx = 2

    max_slice = 64
    x_size = 256
    y_size = 256

    valid_generator_inv = vxm_data_generator(inv_data_path, max_slice, x_size, y_size, filelist_inv, 1, normalize=False,
                                            shuffle=False, start_idx=start_idx)
    val_input_inv, val_output_inv = next(valid_generator_inv)

    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    vol_shape = (max_slice, x_size, y_size)
    vxm_model_inv = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    checkpoint_path_inv = "../model_cp_inv/cp.ckpt"
    vxm_model_inv.load_weights(checkpoint_path_inv)

    val_pred_inv = vxm_model_inv.predict(val_input_inv)

    # Visualization
    images = [img[0, max_slice // 2, :, :, 0] for img in val_input_inv + val_pred_inv]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)


    data_path = "../MRNet/MRNet-v1.0/train/axial"
    filelist = os.listdir(data_path)
    if '.DS_Store' in filelist:
        filelist.remove('.DS_Store')

    valid_generator = vxm_data_generator(data_path, max_slice, x_size, y_size, filelist, 1, normalize=True,
                                         shuffle=False, start_idx=start_idx)
    val_input, val_output = next(valid_generator)

    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    checkpoint_path = "../model_cp/cp.ckpt"
    vxm_model.load_weights(checkpoint_path)

    val_pred = vxm_model.predict(val_input)

    # Visualization
    images = [img[0, max_slice // 2, :, :, 0] for img in val_input + val_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)