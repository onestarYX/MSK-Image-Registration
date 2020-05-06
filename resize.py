
import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

path = './data/'
img1_path = path + 'knee1FS time1 337172.nii.gz'
img2_path = path + 'knee1FS time2 338366.nii.gz'
img3_path = path + 'Knee2NFS Time1 20090305.nii.gz'
img4_path = path + 'Knee2NFS Time2 20140523.nii.gz'

img1Obj = nib.load(img1_path)
img2Obj = nib.load(img2_path)
img3Obj = nib.load(img3_path)
img4Obj = nib.load(img4_path)

im10 = np.expand_dims(img1Obj.get_fdata(),axis=0)
im20 = np.expand_dims(img2Obj.get_fdata(),axis=0)
im30 = np.expand_dims(img3Obj.get_fdata(),axis=0)
im40 = np.expand_dims(img4Obj.get_fdata(),axis=0)

print(im10.shape, im20.shape)
print(im30.shape, im40.shape)

atlas = im10.copy()
X_vol = im20.copy()

atlas_224 = []
for i in range(len(atlas[0])):
    sub = []
    for j in range(len(atlas[0][i])):
        su = np.append(atlas[0][i][j], np.zeros((32,1)))
        sub.append(su)
    atlas_224.append(sub)

X_vol_224 = []
for i in range(len(X_vol[0])):
    sub = []
    for j in range(len(X_vol[0][i])):
        su = np.append(X_vol[0][i][j], np.zeros((32,1)))
        sub.append(su)
    X_vol_224.append(sub)

atlas_224 = np.array(atlas_224)
X_vol_224 = np.array(X_vol_224)
print((atlas_224).shape)
print((X_vol_224).shape)
# plt.imshow(atlas_224[:,:,191])
# plt.show()

atlas_224_resized = atlas_224.copy()
stretch_near = cv2.resize(atlas_224_resized, (192, 160), interpolation = cv2.INTER_NEAREST)
Titles ="Interpolation Nearest"
atlas_224_resized = stretch_near
plt.imshow(atlas_224_resized[:,:,200])
plt.show()

X_vol_224_resized = X_vol_224.copy()
stretch_near = cv2.resize(X_vol_224_resized, (192, 160), interpolation = cv2.INTER_NEAREST)
Titles ="Interpolation Nearest"
X_vol_224_resized = stretch_near

X_vol_224_resized = np.expand_dims(np.expand_dims(X_vol_224_resized, 0), -1)
atlas_224_resized = np.expand_dims(np.expand_dims(atlas_224_resized, 0), -1)

print(X_vol_224_resized.shape)
print(atlas_224_resized.shape)
