import os, sys
import numpy as np
import random

def pn_data_generator(data_path, max_slice, x_size, y_size, filelist, batch_size=1, normalize=False, shuffle=True,
                      start_idx=0):
    """
    Generator that yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """
    i = start_idx
    while True:
        if shuffle:
            fixed_img_paths = random.choices(filelist, k=batch_size)
            moving_img_paths = random.choices(filelist, k=batch_size)
        else:
            if i + batch_size * 2 > len(filelist) - 1:
                i = 0
            fixed_img_paths = filelist[i : i + batch_size]
            moving_img_paths = filelist[i + batch_size : i + batch_size * 2]
            i += batch_size * 2

        fixed_img_list = []
        for fixed_img_path in fixed_img_paths:
            fixed_img = np.load(os.path.join(data_path, fixed_img_path))
            # fixed_img = zoom(fixed_img, (2, 1, 1))
            num_slice_to_pad = max_slice - fixed_img.shape[0]
            left_padding = num_slice_to_pad // 2
            right_padding = num_slice_to_pad - left_padding
            # print("fixed left_padding: {}".format(left_padding))
            # print("fixed right_padding: {}".format(right_padding))
            fixed_img = np.pad(fixed_img, ((left_padding, right_padding), (0, 0), (0, 0)), 'constant')
            fixed_img_list.append(fixed_img)

        moving_img_list = []
        for moving_img_path in moving_img_paths:
            moving_img = np.load(os.path.join(data_path, moving_img_path))
            # moving_img = zoom(moving_img, (2, 1, 1))
            num_slice_to_pad = max_slice - moving_img.shape[0]
            left_padding = num_slice_to_pad // 2
            right_padding = num_slice_to_pad - left_padding
            # print("moving left_padding: {}".format(left_padding))
            # print("moving right_padding: {}".format(right_padding))
            moving_img = np.pad(moving_img, ((left_padding, right_padding), (0, 0), (0, 0)), 'constant')
            moving_img_list.append(moving_img)

        fixed_images = np.array(fixed_img_list)
        if normalize:
            fixed_images = fixed_images / 255.0
        fixed_images = np.expand_dims(fixed_images, -1)
        moving_images = np.array(moving_img_list)
        if normalize:
            moving_images = moving_images / 255.0
        moving_images = np.expand_dims(moving_images, -1)

        vol_shape = (max_slice, x_size, y_size)
        ndims = len(vol_shape)

        # prepare a zero array the size of the deformation
        zero_phi = np.zeros([batch_size, *vol_shape, ndims])

        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]

        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)