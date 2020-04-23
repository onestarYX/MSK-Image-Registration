import os
import numpy as np
import nibabel as nib

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


imgList = []
imgList.append(img1)
imgList.append(img2)
imgList.append(img3)
imgList.append(img4)

# Write the img array to a txt file (slow and huge, caution!)
file = open("./data/NetworkInput/sampleKneeInput.txt", "w")
for img in imgList:
    for sli in range(0, img.shape[2]):
        for row in range(0, img.shape[0]):
            for col in range(0, img.shape[1]):
                file.write(str(img[row][col][sli]) + " ")
            file.write('\n')
