import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data_path = "../MRNet/MRNet-v1.0/train/axial"
output_path = "../MRNet/MRNet-v1.0/train/axial_inv"
os.makedirs(output_path, exist_ok=True)

if __name__ == "__main__":
    filelist = os.listdir(data_path)
    filelist.remove('.DS_Store')

    for file_name in tqdm(filelist):
        image_path = os.path.join(data_path, file_name)
        img = np.load(image_path)

        img_inverse = np.copy(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    img_inverse[i, j, k] = 255 - img_inverse[i, j, k]

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # From left to right
                for k in range(img.shape[2]):
                    if img_inverse[i, j, k] < 0.8:
                        break;
                    img_inverse[i, j, k] = 0
                # From right to left
                for k in range(img.shape[2] - 1, -1, -1):
                    if img_inverse[i, j, k] < 0.8:
                        break;
                    img_inverse[i, j, k] = 0

        np.save(os.join.path(output_path, file_name), img_inverse)