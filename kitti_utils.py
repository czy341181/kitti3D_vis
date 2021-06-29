""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import cv2
import os

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'] 
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)
        self.path = calib_filepath
    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0];
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1];
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2];
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)
 
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)    

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
    #print 'cornsers_3d: ', corners_3d 
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''
    
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
   
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])
    
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]
    
    # vector behind image plane?
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, np.transpose(orientation_3d)
    
    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image


class Projector:
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |fc2 278969
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    '''
    bev_geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'input_shape': (800, 700, 3),
        'label_shape': (200, 175, 7)
    }

    def __init__(self, calib_path=None, calib=None):
        self.calib = calib if calib is not None else Calib(calib_path)
        self.P = self.calib.P
        self.R = self.calib.R
        self.T = self.calib.T
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (- self.f_u)  # relative
        self.b_y = self.P[1, 3] / (- self.f_v)

    ###############integration method#######################
    def project2img(self, points, canvas):
        '''
        project world points to 2d canvas
        return canvas
        '''
        # img = pic.copy()
        uvZ, mask = self.world2img(points)
        uvZ = uvZ[mask, :]
        (h, w) = canvas.shape
        inRangeW = self.inrange(uvZ[:, 0], 0, w)
        inRangeH = self.inrange(uvZ[:, 1], 0, h)
        inRange = np.logical_and(inRangeH, inRangeW)
        uvZ = uvZ[inRange, :]
        for i in range(uvZ.shape[0]):
            canvas[int(round(uvZ[i, 1])), int(round(uvZ[i, 0]))] = round(uvZ[i, 2] * 256.0)
            # img[int(round(coor[i, 1])), int(round(coor[i, 0])), 1] = 0
            # img[int(round(coor[i, 1])), int(round(coor[i, 0])), 2] = 0
        return canvas, mask

    def project2velo(self, img, real_lidar=None, sample_rate=1):
        uv = self.generate_index_matrix(img.shape)
        Z = img[..., np.newaxis] / 256.
        uvZ = np.concatenate((uv, Z), 2).reshape(-1, 3)
        xyz = self.img2world(uvZ)
        if real_lidar is None:
            i = np.zeros((xyz.shape[0], 1))
            xyzi = np.concatenate((xyz, i), 1)
        else:
            mark = np.zeros((xyz.shape[0], 1))
            real_lidar = real_lidar.reshape(-1, 1)
            mark[real_lidar > 0.001] = 1
            # print(sum(mark))
            xyzi = np.concatenate((xyz, mark), 1)
        return xyzi

    def project2bev(self, points, canvas, color=255):
        "https://github.com/ankita-kalra/PIXOR/blob/master/datagen.py"
        if canvas is None:
            canvas = np.zeros(self.bev_geometry['input_shape'])
        for i in range(points.shape[0]):
            if self.point_in_roi(points[i, :]):
                x = int((points[i, 1] - self.bev_geometry['L1']) / 0.1)
                y = int((points[i, 0] - self.bev_geometry['W1']) / 0.1)
                if 0 <= x < canvas.shape[0] and 0 <= y < canvas.shape[1]:
                    canvas[x, y, :] = color
        return canvas

    def world2img(self, points):
        """
            in: N*[x y z]
            out: N*[u v Z_cam2], mask N*[bool] (befor camera)
        """
        x_ref, mask = self.world2ref(points.transpose())
        x_rect = self.ref2rect(x_ref)
        x_img = self.rect2img(x_rect).transpose()
        # x_img[:, :2] = np.round(x_img[:, :2])
        return x_img, mask

    def img2world(self, uvZ):
        x_cam2 = self.img2cam2(uvZ.transpose())
        x_rect = self.cam22rect(x_cam2)
        x_ref = self.rect2ref(x_rect)
        return self.ref2world(x_ref).transpose()

    #################projection tool
    def inverse(self, m):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv = np.linalg.inv(m)
        return inv

    def cart2homv(self, points):
        return np.vstack((points, np.ones((1, points.shape[1]))))

    def cart2homh(self, points):
        return np.hstack((points, np.ones((points.shape[0], 1))))

    def inrange(self, x, low, up):
        if up < low:
            up, low = low, up
        return np.logical_and(x >= low, x < up - 1)

    def point_in_roi(self, point):
        "https://github.com/ankita-kalra/PIXOR/blob/master/datagen.py"
        if (point[0] - self.bev_geometry['W1']) < 0.01 or (self.bev_geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.bev_geometry['L1']) < 0.01 or (self.bev_geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.bev_geometry['H1']) < 0.01 or (self.bev_geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def generate_index_matrix(self, img_size):
        '''
        compute matrix N*M*2 to perform pixel's coordinate on image
        output [Y, X, 2]
        '''
        x = np.arange(img_size[1])
        x = x[np.newaxis, :].repeat(img_size[0], 0)
        y = np.arange(img_size[0])
        y = y[:, np.newaxis].repeat(img_size[1], 1)

        return np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), 2)

    ##############velo 2 img####################
    def world2ref(self, x_world):
        x_world_hom = self.cart2homv(x_world)
        x_ref = np.dot(self.T, x_world_hom)
        return x_ref, x_ref[2, :] > 0

    def ref2rect(self, x_ref):
        R0 = np.eye(4)
        R0[:3, :3] = self.R
        return np.dot(R0, x_ref)

    def rect2img(self, x_rect):
        """
            return N*[u, v, Z_cam2]
        """
        x_img = np.dot(self.P, x_rect)
        x_img[0, :] /= x_img[2, :]
        x_img[1, :] /= x_img[2, :]
        return x_img

    ##############img 2 velo#####################
    def img2cam2(self, UVZ):
        uvZ = UVZ.copy()
        uvZ[0, :] *= uvZ[2, :]
        uvZ[1, :] *= uvZ[2, :]
        uvZ_hom = self.cart2homv(uvZ)
        P_ = np.eye(4)
        P_[0, 2] = - self.c_u
        P_[0, :] /= self.f_u
        P_[1, 2] = - self.c_v
        P_[1, :] /= self.f_v
        '''
        P_ inv P
        pc = P_.copy()s
        R = np.eye(4)
        P_[0, 3] = self.b_x
        P_[1, 3] = self.b_y
        P2 = np.eye(4)
        P2[:3,:] = self.P
        print(np.dot(P2, P_ ))
        '''
        return np.dot(P_, uvZ_hom)

    def cam22rect(self, x_cam2):
        R = np.eye(4)
        R[0, 3] = self.b_x
        R[1, 3] = self.b_y
        return np.dot(R, x_cam2)

    def rect2ref(self, x_rect):
        R_ = np.eye(4)
        R_[:3, :3] = self.inverse(self.R)
        return np.dot(R_, x_rect)

    def ref2world(self, x_ref):
        T_ = self.inverse(self.T)
        return np.dot(T_, x_ref)[:3, :]


class extractor:
    '''
    compute on cpu
    extract img pixels in given 3d box
    '''

    def __init__(self, img_size=(400, 1300)):
        self.idx_matrix = self.generate_index_matrix(img_size)

        #######################tool###########################

    def generate_index_matrix(self, img_size):
        '''
        compute matrix N*M*2 to perform pixel's coordinate on image
        output [Y, X, 2]
        '''
        x = np.arange(img_size[1])
        x = x[np.newaxis, :].repeat(img_size[0], 0)
        y = np.arange(img_size[0])
        y = y[:, np.newaxis].repeat(img_size[1], 1)

        return np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), 2)

    #####################extract function################`
    def extract_3dpoints_in_3dbox(self, points, dddbox, projector):
        '''
        params: points (N, 3)
                3dbox [x,y,z,h,w,l,o (orientaion)](in camera coodinate)
        rotat axis first
        '''
        rotat = rotat_z(dddbox[6])
        box_xyz = np.ones((4, 1))
        box_xyz[0:3, 0] = dddbox[0:3]
        world_box = projector.ref2world(projector.rect2ref(box_xyz))
        world_box = np.dot(rotat, world_box)

        '''
        dddbox_new = dddbox[[2, 0, 1]] * np.array([-1,-1,1])
        dddbox_new = np.dot(rotat ,dddbox_new[:, np.newaxis])
        '''
        dddbox_corner1 = world_box[:3, 0] + [*(dddbox[[4, 5]] / 1.7), 0]
        dddbox_corner2 = world_box[:3, 0] - [*(dddbox[[4, 5]] / 1.7), 0]
        dddbox_corner1[2] += dddbox[3]

        rotat_points = np.dot(rotat, points.transpose())
        # [3, N]
        mask1 = (dddbox_corner1[:, np.newaxis] > rotat_points).all(axis=0)
        mask2 = (rotat_points > dddbox_corner2[:, np.newaxis]).all(axis=0)
        return np.logical_and(mask1, mask2)

    def extract_2dpoints_in_3dbox(self, dep, lu, rd, dddbox, projector):
        # print(lu,rd)
        rd[0] = min(rd[0], 1224)
        rd[1] = min(rd[1], 370)
        uv = np.copy(self.idx_matrix[lu[1]:rd[1], lu[0]:rd[0], :]).astype(float)
        Z = np.copy(dep[lu[1]:rd[1], lu[0]:rd[0], np.newaxis])
        uvZ = np.concatenate((uv, Z), 2).reshape(-1, 3)
        uvZ[:, 2] /= 256.0

        # print(uvZ)
        points_world = projector.img2world(uvZ)
        # print(points_world.max(axis=0))
        mask = self.extract_3dpoints_in_3dbox(points_world, dddbox, projector)
        return mask.reshape(rd[1] - lu[1], rd[0] - lu[0])

    def extract_fov_points(self, pc, bbox, projector):
        # print(lu,rd)
        uvZ, mask_front = projector.world2img(pc[:, :3])

        # sao cao zuo
        mask = np.zeros((pc.shape[0]), dtype=bool)

        [xmin, ymin, xmax, ymax] = list(bbox)

        box_fov_inds = (uvZ[:, 0] < xmax) & \
                       (uvZ[:, 0] >= xmin) & \
                       (uvZ[:, 1] < ymax) & \
                       (uvZ[:, 1] >= ymin) & mask_front

        mask[box_fov_inds] = True

        # print(mask.sum(), pc_front_mask.sum(), mask.shape(), pc_front_mask.shape)

        return mask

    def extract_seg_part_points(self, dep, bbox, seg, projector):
        [l, u, w, h] = bbox
        r = min(l + w, 1224)
        d = min(u + h, 370)
        uv = np.copy(self.idx_matrix[u:d, l:r, :].astype(float)
        Z = np.copy(dep[u:d, l:r, :])
        S = np.copy(seg[u:d, l:r, :])
        uvZ = np.concatenate((uv, Z), 2).reshape(-1, 3)
        S = S.reshape(-1, 1)
        uvZ = uvZ(S > 0.1)
        uvZ[:, 2] /= 256.0

        return uvZ

