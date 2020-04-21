import os
import numpy as np
import nibabel as nib

img1_path = 'data/knee1FS time1 337172.nii.gz'
img2_path = 'data/knee1FS time2 338366.nii.gz'
img1 = nib.load(img1_path)
img2 = nib.load(img2_path)

print(img1.shape)
print(img2.shape)
