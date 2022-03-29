import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

data_path = "../MRNet/MRNet-v1.0/train/axial"
output_path = "../MRNet/MRNet-v1.0/train/axial_inv"
os.makedirs(output_path, exist_ok=True)

if __name__ == "__main__":
    filelist = os.listdir(data_path)
    filelist.remove('.DS_Store')

    for file_name in tqdm(filelist):
        image_path = os.path.join(data_path, file_name)
        img = np.load(image_path)

        img_denoised = np.zeros_like(img)
        for i in range(img.shape[0]):
            img_denoised[i] = cv2.fastNlMeansDenoising(img[i], None, 10)

        mask = np.ones_like(img_denoised)
        for i in range(img_denoised.shape[0]):
            for j in range(img_denoised.shape[1]):
                # From left to right
                for k in range(img_denoised.shape[2]):
                    if img_denoised[i, j, k] < 0.2 * 255:
                        mask[i, j, k] = 0
                    else:
                        break
                # From right to left
                for k in range(img_denoised.shape[2] - 1, -1, -1):
                    if img_denoised[i, j, k] < 0.2 * 255:
                        mask[i, j, k] = 0
                    else:
                        break

        img_inverse = np.copy(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    img_inverse[i, j, k] = 255 - img_inverse[i, j, k]

        img_processed = img_inverse * mask
        img_processed = img_processed.astype('float32')
        img_processed /= 255.0

        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(img[20, :, :])
        # axs[1].imshow(img_processed[20, :, :])
        # plt.show()

        np.save(os.path.join(output_path, file_name), img_processed)