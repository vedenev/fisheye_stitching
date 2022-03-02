#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:28:43 2022

@author: vedenev
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt 

DEBUG_PLOT = True

IMAGES_DIR = 'images'

file_1 = IMAGES_DIR + '/' + 'camera_1.jpg'
file_2 = IMAGES_DIR + '/' + 'camera_2.jpg'

SHIFT_1_X = -300
SHIFT_1_Y = 0


DIM = (2560, 1920) # input video dimetions, (width, height)
K = np.array([[660.6883514485156, 0.0, 1294.1580197345322], [0.0, 661.8183036998079, 985.5144341662063], [0.0, 0.0, 1.0]]) # final camera marix constants (focus and shifts)
D = np.array([[-0.01393423897472937], [-0.009833220533829656], [0.006403434693351997], [-0.0018652582823445753]]) # camera distortion coefficients

h_celing = 5.3 # meight of the camera (= height of the celing) in meters

def meshgrid_maps_float32(shape_t):
    # return meshgrid maps in float32 format
    # input: shape_t=(height, widht) - shape request for meshgrid mapms
    # output: X_pixels_f32, Y_pixels_f32 - meshgrid maps
    x_pixels = np.arange(shape_t[1])
    y_pixels = np.arange(shape_t[0])
    X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels)
    X_pixels_f32 = X_pixels.astype(np.float32)
    Y_pixels_f32 = Y_pixels.astype(np.float32)
    
    return X_pixels_f32, Y_pixels_f32

def imshow_rgb(image_bgr):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image)


def get_mask_contour(point_1, point_2, height, width):
    mask = np.zeros((height, width), np.uint8)
    vector = point_1 - point_2
    vector = vector / np.sqrt(np.sum(vector**2))
    center = (point_1 + point_2) / 2
    vector = vector.flatten()
    center = center.flatten()
    x = np.arange(width).astype(np.float32)
    y = np.arange(height).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    dX = X - center[0]
    dY = Y - center[1]
    prod = dX * vector[0] + dY * vector[1]
    condition = prod >= 0
    mask[condition] = 255
    
    contours, hierarchy = cv2.findContours(mask.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    
    # reduce number of vertices to 4
    peri = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.05 * peri, True)
    vertices = vertices[:, 0, :]
    
    #print(vertices.shape)
    #plt.imshow(mask)
    #import sys
    #sys.exit()
    
   
    
    return vertices, mask

def plot_cotour(contour):
    contour_2 = np.concatenate((contour, contour[-1:,:]), axis=0)
    plt.plot(contour_2[:, 0], contour_2[:, 1], 'b.-')


def floor_to_fisheye(X_floor, Y_floor):
    
    global h_celing
    global D
    global K
    
    Z_floor = h_celing * np.ones_like(X_floor) # distance between camera and floow
    
    # X_floor, Y_floor, Z_floor - coodrinatis of floor plane relative to camera in meters
    
    X = X_floor
    Y = Y_floor
    Z = Z_floor
    
    
    a = X / Z
    b = Y / Z
    r = np.sqrt(a**2 + b**2)
    theta = np.arctan(r)
    theta_d = theta * (1.0 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)
    
    xs = theta_d * a / r
    ys = theta_d * b / r
    
    x = K[0,0]*xs + K[0,2]
    y = K[1,1]*ys + K[1,2]
    
    x[Z<0] = 0.0
    y[Z<0] = 0.0
    
    # to insure that it is float32 type:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    return x, y


def combine(image_1, image_2, mask_1):
    mask_1_f32 = mask_1.astype(np.float32) / 255.0
    mask_2_f32 = 1 - mask_1_f32
    image_1_f32 = image_1.astype(np.float32)
    image_2_f32 = image_2.astype(np.float32)
    combined_f32 = np.zeros_like(image_1_f32)
    combined_f32[:, :, 0] = mask_1_f32 * image_1_f32[:, :, 0] + mask_2_f32 * image_2_f32[:, :, 0]
    combined_f32[:, :, 1] = mask_1_f32 * image_1_f32[:, :, 1] + mask_2_f32 * image_2_f32[:, :, 1]
    combined_f32[:, :, 2] = mask_1_f32 * image_1_f32[:, :, 2] + mask_2_f32 * image_2_f32[:, :, 2]
    combined = combined_f32.astype(np.uint8)
    return combined



# here: central part of the video will be rectified
# resolution of the output video
h = 1500 # height
w = 1500 # widht
# h == w

field_size = 50.0 # central fild size, in meters, field_size x field_size

#angle_rotation_degrees = 40.0 # rotate view in horizontal plane at this angle, in degrees
#angle_rotation = np.pi * angle_rotation_degrees / 180.0

X_floor_1_0, Y_floor_1_0 = meshgrid_maps_float32((h, w))
# nirmalize and shift to center:
X_floor_1 = field_size * (X_floor_1_0 - X_floor_1_0[h//2, w //2] + SHIFT_1_X) / h
Y_floor_1 = field_size * (Y_floor_1_0 - Y_floor_1_0[h//2, w //2] + SHIFT_1_Y) / h

x1, y1 = floor_to_fisheye(X_floor_1, Y_floor_1)



X_floor_2_0, Y_floor_2_0 = meshgrid_maps_float32((h, w))

# rotate at 90 gerees:
X_floor_2_1 = -(Y_floor_2_0 - Y_floor_2_0[h//2, w //2]) + X_floor_2_0[h//2, w //2]
Y_floor_2_1 = (X_floor_2_0 - X_floor_2_0[h//2, w //2]) + Y_floor_2_0[h//2, w //2]

X_floor_2 = field_size * (X_floor_2_1 - X_floor_2_0[h//2, w //2]) / h
Y_floor_2 = field_size * (Y_floor_2_1 - Y_floor_2_0[h//2, w //2]) / h

x2, y2 = floor_to_fisheye(X_floor_2, Y_floor_2)


points_1 = np.asarray([[419.5, 724.4],
                       [351.5, 424.5],
                       [533.1, 665.5],
                       [367.1, 913.6]], np.float32)
    
points_1[:, 0] -= SHIFT_1_X
points_1[:, 1] -= SHIFT_1_Y
    
points_2 = np.asarray([[893.4, 754.4],
                       [967.7, 556.1],
                       [1007.7, 755.7],
                       [734.0, 887.9]], np.float32)


frame_1 = cv2.imread(file_1)
frame_1_unroled = cv2.remap(frame_1, x1, y1, cv2.INTER_CUBIC) 



frame_2 = cv2.imread(file_2)
frame_2_unroled = cv2.remap(frame_2, x2, y2, cv2.INTER_CUBIC)


H = cv2.getPerspectiveTransform(points_2, points_1)
H_invereted = cv2.getPerspectiveTransform(points_1, points_2)

center_2_orig_x = w / 2
center_2_orig_y = h / 2

center_2_orig_uniform = np.asarray([[center_2_orig_x],
                                    [center_2_orig_y],
                                    [1]])
center_2_orig = np.asarray([[center_2_orig_x], 
                            [center_2_orig_y]])

center_2_uniform = np.matmul(H, center_2_orig_uniform)
center_2 = center_2_uniform[0:2] / center_2_uniform[2]

center_1_orig_x = w / 2
center_1_orig_y = h / 2

center_1_orig_x -= SHIFT_1_X
center_1_orig_y -= SHIFT_1_Y


center_1_orig = np.asarray([[center_1_orig_x], 
                            [center_1_orig_y]])
center_1_orig_uniform = np.asarray([[center_1_orig_x],
                                    [center_1_orig_y],
                                    [1]])
    
center_1_uniform = np.matmul(H_invereted, center_1_orig_uniform)
center_1 = center_1_uniform[0:2] / center_1_uniform[2]


mask_contour_1, mask_1 = get_mask_contour(center_1_orig, center_2, h, w)
mask_contour_2, _ = get_mask_contour(center_2_orig, center_1, h, w)



X_floor_2_rotated_0, Y_floor_2_rotated_0 = meshgrid_maps_float32((h, w))


denumerator = H_invereted[2, 0] * X_floor_2_rotated_0 + H_invereted[2, 1] * Y_floor_2_rotated_0 + H_invereted[2, 2]
X_floor_2_rotated_1 = (H_invereted[0, 0] * X_floor_2_rotated_0 + H_invereted[0, 1] * Y_floor_2_rotated_0 + H_invereted[0, 2]) / denumerator
Y_floor_2_rotated_1 = (H_invereted[1, 0] * X_floor_2_rotated_0 + H_invereted[1, 1] * Y_floor_2_rotated_0 + H_invereted[1, 2]) / denumerator

# rotate at 90 gerees:
X_floor_2_rotated_2 = -(Y_floor_2_rotated_1 - Y_floor_2_rotated_0[h//2, w //2]) + X_floor_2_rotated_0[h//2, w //2]
Y_floor_2_rotated_2 = (X_floor_2_rotated_1 - X_floor_2_rotated_0[h//2, w //2]) + Y_floor_2_rotated_0[h//2, w //2]



# normalize and shift to center:
X_floor_2_rotated = field_size * (X_floor_2_rotated_2 - X_floor_2_rotated_0[h//2, w //2]) / h
Y_floor_2_rotated = field_size * (Y_floor_2_rotated_2 - Y_floor_2_rotated_0[h//2, w //2]) / h

x2_rotated, y2_rotated = floor_to_fisheye(X_floor_2_rotated, Y_floor_2_rotated)


frame_2_unroled_rotated = cv2.remap(frame_2, x2_rotated, y2_rotated, cv2.INTER_CUBIC)


combined = combine(frame_1_unroled, frame_2_unroled_rotated, mask_1)

cv2.imwrite('combined.png', combined)

if DEBUG_PLOT:
    
    plt.close('all')
    
    plt.subplot(2, 2, 1)
    imshow_rgb(frame_1_unroled)
    plt.plot(points_1[:, 0], points_1[:, 1], 'g.-')
    plt.plot(center_2[0], center_2[1], 'mx')
    plt.plot(center_1_orig_x, center_1_orig_y, 'cx')
    plot_cotour(mask_contour_1)
    plt.title('camera 1')
    
    plt.subplot(2, 2, 2)
    imshow_rgb(frame_2_unroled)
    plt.plot(points_2[:, 0], points_2[:, 1], 'g.-')
    plt.plot(center_2_orig_x, center_2_orig_y, 'mx')
    plt.plot(center_1[0], center_1[1], 'cx')
    plot_cotour(mask_contour_2)
    plt.title('camera 2')
    
    
    plt.subplot(2, 2, 3)
    imshow_rgb(frame_2_unroled_rotated)
    plt.title('camera 2 in coordinates of camera 1') 
    
    plt.subplot(2, 2, 4)
    imshow_rgb(combined)
    plt.title('combined')
    
    
    plt.show()




