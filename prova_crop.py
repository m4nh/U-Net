import cv2
import os
import sys

import glob
import numpy as numpy


path = '/home/daniele/data/datasets/kitti/2011_10_03_drive_0027_sync/image_02/data'
outputfolder = '/home/daniele/data/datasets/kitti/2011_10_03_drive_0027_sync/image_02/cropped'
try:
    os.mkdir(outputfolder)
except:
    pass
files = glob.glob(os.path.join(path, "*.jpg"))

print(files)

manifest = []
cropsize_w = 1024
cropsize_h = 256
for i, f in enumerate(files):
    img = cv2.imread(f)
    h, w, _ = img.shape

    remain = int((w - cropsize_w)*0.5)
    newimg = img[:, remain:remain+cropsize_w]

    remain = int((h - cropsize_h)*0.5)
    newimg = newimg[remain:remain+cropsize_h, :]

    outputfile = os.path.join(outputfolder, os.path.basename(f))
    if os.path.exists(outputfile):
        print("FILE EXISTS! ", outputfile)
        sys.exit(0)
    manifest.append(os.path.abspath(outputfile))
    cv2.imwrite(outputfile, newimg)
    cv2.imshow("img", img)
    cv2.waitKey(1)

manifestfile = os.path.join(outputfolder, "manifest.txt")
out = open(manifestfile, 'w')
for m in manifest:
    out.write("{};{}\n".format(
        m, m
    ))

print(manifest)
