from viz_util import *
import cv2
import os
import numpy as np
import mayavi.mlab as mlab
from PIL import Image
from bev import *


def draw_rgb_3dboxes(reader, rgb_out=None):
    # draw all 3d boxes in reader and save to png and pdf
    for data in reader:
        (idx, rgb, velo, objs, cal) = data
        print(idx)
        #img = draw_rgb_3dbox(rgb, cal, res, (0,0,255))    #pred
        #img = draw_rgb_3dbox(img, cal, res_pred, (255,0,0))    #pred
        img = draw_rgb_3dbox(rgb, cal, objs, (0,255,0))    #gt
        #save
        if rgb_out:
            cv2.imwrite( os.path.join(rgb_out, idx+'.png'), img)
            image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            image.save(os.path.join(rgb_out,'rgb' +  idx+'.pdf'))

def draw_bev(reader, rgb_out=None):
    # draw all 3d boxes in reader and save to png and pdf
    for data in reader:
        (idx, rgb, velo, objs, cal) = data
        print(idx)
        #img = draw_rgb_3dbox(rgb, cal, res, (0,0,255))    #pred
        #img = draw_rgb_3dbox(img, cal, res_pred, (255,0,0))    #pred
        img = draw_box3d_on_bev(velo, cal, objs)
        #img = draw_bev_img(velo, cal, objs)    #gt
        #save
        if rgb_out:
            #cv2.imwrite(rgb_out, img)
            cv2.imwrite( os.path.join(rgb_out, idx+'.png'), img)
        #     image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #     image.save(os.path.join(rgb_out,'rgb' +  idx+'.pdf'))


def draw_rgb_2dboxes(reader, rgb_out=None):
    # draw all 2d boxes in reader and save to png and pdf
    for data in reader:
        (idx, rgb, velo, objs, cal) = data
        print(idx)
        #img = draw_rgb_2dbox(rgb, cal, res, (0,0,255))
        img = draw_rgb_2dbox(rgb, cal, objs, (0,255,0))
        #save
        if rgb_out:
            #cv2.imwrite( os.path.join(rgb_out, idx+'.png'), img)
            image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            image.save(os.path.join(rgb_out, 'rgb' + idx+'.pdf'))

def draw_lidar_3dboxes(reader, out=False):
    # draw 3d boxes of all data in reader, and pop window one by one
    for data in reader:
        (idx, rgb, velo, objs, cal) = data
        print(idx)
        fig = mlab.figure(size=(1200,800), bgcolor=(0.9,0.9,0.85))
        fig = draw_lidar_fig(fig, velo, objs, cal,(0,1,0), drawlidar=True,GT=True)
        #fig = draw_lidar_fig(fig, velo, res_baseline, cal,(1,0,0), drawlidar=False,GT=True)
        #fig = draw_lidar_fig(fig, velo, res_pred, cal,(0,0,1), drawlidar=False,GT=False)
        mlab.view(azimuth=135, elevation=70, focalpoint=[ 22.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
        mlab.show()
        # if out:
        #     mlab.savefig( filename=PATH + '/%06d.pdf'%int(idx))
        #     mlab.close()
        # else:
        #     mlab.show()
    input()

def draw_lidar_3dboxes_test(reader):
    # draw 3d boxes of all data in reader, and pop window one by one
    for data in reader:
        (idx, rgb, velo, objs, res, cal) = data
        print(idx)
        fig = mlab.figure(size=(1200,800), bgcolor=(0.9,0.9,0.85))
        fig = draw_lidar_fig(fig, velo, res, cal,(1,0,0), drawlidar=True, GT=True)
        mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
        mlab.show()
    input()

def draw_lidar_3dboxes_cmp(reader):
    for data in reader:
        (idx, rgb, velo, objs, mono, stereo, cal) = data
        print(idx)
        fig = mlab.figure(size=(1200,800), bgcolor=(0.9,0.9,0.85))
        fig = draw_lidar_fig(fig, velo, objs, cal,(0,1,0), drawlidar=True,lidar_color=0,GT=True)
        fig = draw_lidar_fig(fig, velo, mono, cal,(0,0,1), drawlidar=False)
        fig = draw_lidar_fig(fig, velo, stereo, cal,(0.95,0.42,0), drawlidar=False)
        mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
        mlab.show()
    input()
def draw_rgb_3dbox(img, calib, objs, c = (255,255,255)):
    def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
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
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    for obj in objs:
        #if obj.type not in ['Car']:
        #    continue
        color = c
        if obj.type == 'Car': color = (0, 165, 255)  # for car
        elif obj.type == 'Pedestrian': color = (87, 187, 123)  # for car
        elif obj.type == 'Cyclist': color = (187, 87, 123)  # for car
        #else: continue
        b, ddd = compute_box_3d(obj, calib.P)
        if b is not None:
            img = draw_projected_box3d(img, b, color)
    return img

def draw_rgb_2dbox(img, calib, objs, c = (255,255,255)):
    for obj in objs:
        cv2.line(img, (round(obj.xmin), round(obj.ymin)), (round(obj.xmin), round(obj.ymax)), color=c)
        cv2.line(img, (round(obj.xmin), round(obj.ymin)), (round(obj.xmax), round(obj.ymin)), color=c)
        cv2.line(img, (round(obj.xmax), round(obj.ymax)), (round(obj.xmin), round(obj.ymax)), color=c)
        cv2.line(img, (round(obj.xmax), round(obj.ymax)), (round(obj.xmax), round(obj.ymin)), color=c)
    return img


def draw_box3d_on_bev(
    velo,
    cal,
    objs,
    color=(255, 255, 255),
    thickness=2,
    scores=None,
    text_lables=[],
    is_gt=False,
):


    top_view = lidar_to_top(velo)
    top_image = draw_top_image(top_view)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = top_image.copy()
    num = len(objs)

    def bbox3d(obj):
        _, box3d_pts_3d = compute_box_3d(obj, cal.P)
        box3d_pts_3d_velo = cal.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objs if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    lines = [obj.type for obj in objs if obj.type != "DontCare"]

    top_image = draw_box3d_on_top(
        img, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )

    return top_image


def draw_lidar_fig(fig, velo, objs, calib, c=(1,1,1), drawlidar=True,lidar_color=None, GT=False):
    if drawlidar:
        fig = draw_lidar_pc(velo, fig=fig, color=lidar_color)
    for obj in objs:
        #if obj.type not in ['Car']:
        #    continue

        color = c
        if obj.type == 'Car': color = (0, 165/255, 255/255)  # for car
        elif obj.type == 'Pedestrian': color = (87/255, 187/255, 123/255) # for car
        elif obj.type == 'Cyclist': color = (187/255, 87/255, 123/255)  # for car
        #else: continue
        b, ddd = compute_box_3d(obj, calib.P)
        ddd = calib.project_ref_to_velo(ddd)
        box_pc, ind = extract_pc_in_box3d(velo, ddd)
        #print(ddd.shape)
        #print(box_pc.shape)
        if GT:
            draw_lidar_pc(box_pc, (1., 0,0), fig, pts_mode='sphere')
        fig = draw_lidar_3dbox([ddd], fig, color)
    return fig

def draw_lidar_3dbox(gt_boxes3d, fig, color=(1,1,1), line_width=1.5, draw_text=True, text_scale=(1,1,1), color_list=None):
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    return fig

def draw_lidar_pc(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=2, pts_mode='point'):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        pts: 'sphere' to draw balls, 'point' to draw points
    '''
    if pc.shape[0]==0:
        return fig
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

    if color is None:
        x = pc[:, 0]  # x position of point
        y = pc[:, 1]  # y position of point
        col = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
        print(col.shape)
    elif pts_mode=='sphere':
        col = color
        #col = np.expand_dims(np.array(color), 0)
        #col = col.repeat(pc.shape[0], 0)
    else:
        col = np.expand_dims(np.array(color), 0)
        col = col.repeat(pc.shape[0], 0)
        print(col.shape)

    if pts_mode=='sphere':
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=col, mode='sphere', scale_factor=0.2, figure=fig)
    else:
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], col, mode='point', colormap='spectral', scale_factor=pts_scale, figure=fig)

    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    #draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig
