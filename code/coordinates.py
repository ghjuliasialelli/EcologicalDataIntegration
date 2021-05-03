#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:35:34 2021

@author: ghjuliasialelli
"""

import matplotlib.pyplot as plt
import numpy as np

import rasterio as rs
from rasterio.plot import show

def get_crs():
    crs = {}
    for img_type, fn in zip(['ACD', 'L8', 'S2', 'NICFI'], ['GAO_ACD_30m_unmasked.tif', 'l8_sabah.tif', 's2_sabah_1.tif', 'quads/1678-1054.tiff']) :
        img = rs.open(r'{}'.format(fn))
        crs[img_type] = str(img.crs)
    return crs

crs = {'ACD': 'EPSG:32650',
 'L8': 'EPSG:4326',
 'S2': 'EPSG:4326',
 'NICFI': 'EPSG:3857'}

offsets = {'ACD': 5,
 'L8': 5,
 'S2': 15}

from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling

def read_tiff(file_name):
    return rs.open(r'{}'.format(file_name))

def plot_tiff(file_name):
    img = rs.open(r'{}'.format(file_name))
    show(img)

def transform_img(src_path, dst_path, dst_crs = crs['ACD']):
    with rs.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rs.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rs.band(src, i),
                    destination=rs.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

from rasterio.merge import merge
from rasterio.warp import transform_bounds


def write_to_tiff(array, fn, count, dtype, crs, transform):
    height, width = array.shape[1], array.shape[2]
    new_dataset = rs.open(fn, 'w', driver='GTiff', height=height, width=width,
                          count=count, dtype=dtype, crs=crs, transform=transform)
    new_dataset.write(array)
    new_dataset.close()

def merge_tiff(src_files_to_mosaic):
    mos, out_trans = merge(src_files_to_mosaic)
    return mos, out_trans


def ACD_patch(plot = False, save = False):
    ACD = read_tiff('GAO_ACD_30m_unmasked.tif')
    ACD_patch = ACD.read()[0, 6000:6000+5000, 5000:5000+5000]
    if plot : 
        plt.imshow(ACD_patch)
    if save:
        assert(ACD_patch.shape == (5000, 5000))
        np.save('ACD_patch.npy', ACD_patch)
    return ACD_patch
 
def ACD_geo_bounds():
    ACD = read_tiff('GAO_ACD_30m_unmasked.tif')
    ul_x, ul_y = ACD.xy(6000, 5000, offset='ul')
    lr_x, lr_y = ACD.xy(6000+5000, 5000+5000, offset='ul')
    return ul_x, lr_y, lr_x, ul_y # 421740.0, 490380.0, 571740.0, 640380.0
    
def s2_patch(plot = False, save = False):
    ul_i, lr_j, lr_i, ul_j = ACD_geo_bounds()
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:4326', ul_i, lr_j, lr_i, ul_j)
    with rs.open('Sabah_median_patch1_2016.tif') as s2 :
        row_start, col_start = s2.index(ul_x, ul_y)
        row_stop, col_stop = row_start + 15000, col_start + 15000
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        s2_window = s2.read([1,2,3,4], window = window)
        if plot : 
            plt.imshow(s2.read(1, window = window))
        if save : 
            assert(s2_window.shape == (4, 15000, 15000))
            np.save('S2_patch.npy', s2_window)
    return s2_window

def l8_patch(plot = False, save = False):
    ul_i, lr_j, lr_i, ul_j = ACD_geo_bounds()
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:4326', ul_i, lr_j, lr_i, ul_j)
    with rs.open('l8_sabah.tif') as l8 :
        row_start, col_start = l8.index(ul_x, ul_y)
        row_stop, col_stop = row_start + 5000, col_start + 5000
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        l8_window = l8.read([1,2,3,4], window = window)
        if plot : 
            plt.imshow(l8.read(1, window = window))
        if save : 
            assert(l8_window.shape == (4, 5000, 5000))
            np.save('L8_patch.npy', l8_window)
    return l8_window


def generate_windows(img_fn):
    img = np.load(img_fn)
    img_type = img_fn.split('_')[0]
    
    offset = offsets[img_type]
    
    if len(img.shape) == 2 : 
        s1, s2 = img.shape[0], img.shape[1]
        img = img.reshape((1, s1, s2))
    
    x_bound, y_bound = img.shape[1], img.shape[2]
    print(x_bound, y_bound)
    
    res = []
    
    x_start, y_start = 0, 0
    x_stop, y_stop = x_start + offset, y_start + offset
    
    while x_stop <= x_bound and y_stop <= y_bound : 
        
        res.append(img[:, x_start:x_stop, y_start:y_stop])
        
        x_start = x_stop
        x_stop = x_stop + offset
        y_start = y_stop
        y_stop = y_stop + offset
    
    assert(len(res) == 1000)
    return np.array(res)




# methods to call on the image :
# .crs :  The dataset's coordinate reference system
# .count : the number of raster bands in the dataset
# .descriptions : descriptions for each dataset band
# .res : (width, height) of pixels in the units of its coordinate reference system.
# .height
# .width
# window(self, left, bottom, right, top, precision=None)
#      Get the window corresponding to the bounding coordinates.
# xy(self, row, col, offset='center')
#      Returns the coordinates ``(x, y)`` of a pixel at `row` and `col`.
#      The pixel's center is returned by default, but a corner can be returned
#      by setting `offset` to one of `ul, ur, ll, lr`.
# get_gcps(...)
#      Get GCPs and their associated CRS.f
# get_transform(...)
#      Returns a GDAL geotransform in its native form.
# read_crs(...)
#      Return the GDAL dataset's stored CRS
# read_transform(...)
#      Return the stored GDAL GeoTransform
# read(indexes, window)
#   indexes : list of ints or a single int, optional
#   window : a pair (tuple) of pairs of ints or Window, optional
# A dataset’s transform is an affine transformation matrix that maps pixel locations 
# in (row, col) coordinates to (x, y) spatial positions. The product of this matrix 
# and (0, 0), the row and column coordinates of the upper left corner of the dataset, 
# is the spatial position of the upper left corner.
#   dataset.transform * (0, 0)
#   dataset.transform * (dataset.width, dataset.height) : position of the lower right corner
# getting the array indices corresponding to points in georeferenced space
#   x, y = (dataset.bounds.left + 100000, dataset.bounds.top - 50000)
#   row, col = dataset.index(x, y) --> (1666, 3333)
#   band1[row, col] --> some value
# To get the spatial coordinates of a pixel
#   dataset.xy(dataset.height // 2, dataset.width // 2) --> (476550.0, 4149150.0) 



# rasterio.windows.from_bounds(left, bottom, right, top)
#   Get the window corresponding to the bounding coordinates.
# rasterio.warp.transform_bounds(src_crs, dst_crs, left, bottom, right, top, densify_pts=21)
#       Transform bounds from src_crs to dst_crs.
#   e.g. transform_bounds('EPSG:4326', 'EPSG:32650', 116.0, 4.5, 116.5, 5.0)
#       = (389060.17412816663, 497414.4540920852, 444572.12912697124, 552748.6209151235)

# img_l8.window(116.0, 4.5, 116.5, 5.0)