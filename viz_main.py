import pathlib
import cv2
import os
import numpy as np
import mayavi.mlab as mlab
from PIL import Image
from objects import *
from viz_util import *
from draw_util import *


f = open("C:/Users/80712/Desktop/data/KITTI/trainval.txt",'r')
idxlist = []
for file in f.readlines():
    idxlist.append(int(file))

clslist = ['Car','Pedestrian','Cyclist']
rgb_dir = pathlib.Path('C:/Users/80712/Desktop/data/KITTI/training/image_2')
label_dir = pathlib.Path('C:/Users/80712/Desktop/data/KITTI/training/label_2')
calib_dir = pathlib.Path('C:/Users/80712/Desktop/data/KITTI/training/calib')
velo_dir = pathlib.Path('C:/Users/80712/Desktop/data/KITTI/training/velodyne')

#res_dir_pred = pathlib.Path('C:/Users/czy/Desktop/no_car_150')
rgb_out = pathlib.Path('C:/Users/80712/Desktop/data/KITTI/output')

def read_data(idx, read_gt=True):
    rgb = cv2.imread(rgb_dir.absolute().__str__() +'/' + idx + '.png')
    if read_gt:
        objs = read_label(label_dir  / ( idx + '.txt'))
    else:
        objs = []
    objs = [x for x in objs if x.type in clslist]
    velo = np.fromfile(velo_dir.absolute().__str__() + '/' + idx + '.bin', dtype=np.float32).reshape(-1,4)
    #res_baseline = read_label(res_dir_baseline  / ( idx + '.txt'))
    #res_pred = read_label(res_dir_pred  / ( idx + '.txt'))

    cal = Calibration(calib_dir  / ( idx + '.txt'))
    return idx, rgb, velo, objs, cal

def dataReader(read_data=read_data, read_gt=True):
    results = ['%06d.txt'%x for x in idxlist]
    results.sort()
    for r in results:
        #print(r)  000000.txt
        yield read_data(r[:-4], read_gt)

# if __name__=='__main__ draw lidar pc':
#     l = np.fromfile('007480.bin', dtype=np.float32).reshape(-1,4)
#     fig = mlab.figure(size=(1200, 800), bgcolor=(0.9, 0.9, 0.85))
#     fig = draw_lidar_pc(l, fig=fig)
#     mlab.show()

if __name__=='__main__':# draw boxes in 3d space':
    reader = dataReader(read_gt=True)
    draw_lidar_3dboxes(reader)

if __name__=='__main__:# draw 3d boxes in rgb':
    reader = dataReader()
    draw_bev(reader, rgb_out)

if __name__=='__main__:# draw 3d boxes in rgb':
    reader = dataReader()
    draw_rgb_3dboxes(reader, rgb_out)

if __name__=='__main__:# draw 2d detection boxes in rgb':
    reader = dataReader()
    draw_rgb_2dboxes(reader, rgb_out)


