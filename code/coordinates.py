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
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from os import listdir

# i had to use gdal_merge.py for some reason

def get_info():
    crs = {}
    res = {}
    for img_type, fn in zip(['ACD', 'L8', 'S2', 'NICFI'], ['GAO_ACD_30m_unmasked.tif', 'l8_sabah.tif', 's2_sabah_1.tif', 'quads/1678-1054.tiff']) :
        img = rs.open(r'{}'.format(fn))
        crs[img_type] = str(img.crs)
        res[img_type] = img.res[0]
    return crs, res

crs = {'ACD': 'EPSG:32650',
 'L8': 'EPSG:4326',
 'S2': 'EPSG:4326',
 'NICFI': 'EPSG:3857'}

offsets = {'ACD': 5,
 'L8': 5,
 'S2': 15,
 'NICFI' : 30}

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return (array - array_min) / (array_max - array_min)

def replace_nan(arr):
    col_mean = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    out_arr = arr.copy()
    out_arr[inds] = np.take(col_mean, inds[1])
    return out_arr

def pre_processing(arr):
    if len(arr.shape) == 2 :
        return replace_nan(normalize(arr))
    else:
        res = np.empty(arr.shape)
        for i in range(arr.shape[0]):
            res[i,] = replace_nan(normalize(arr[i,]))
        return res


def read_tiff(file_name):
    return rs.open(r'{}'.format(file_name))

def plot_tiff(file_name):
    img = read_tiff(file_name)
    show(img)


def write_to_tiff(array, fn, height, width, count, dtype, crs, transform):
    new_dataset = rs.open(fn, 'w', driver='GTiff', height=height, width=width,
                          count=count, dtype=dtype, crs=crs, transform=transform)
    new_dataset.write(array)
    new_dataset.close()


def ACD_patch(plot = False, save = False, tiff = True):
    ACD = read_tiff('GAO_ACD_30m_unmasked.tif')
    ACD_patch = ACD.read()[0, 6000:6000+5000, 5000:5000+5000]
    ACD_patch = pre_processing(ACD_patch)
    assert(ACD_patch.shape == (5000, 5000))
    ACD_patch = ACD_patch.reshape((1, 5000, 5000))
    
    if plot : plt.imshow(ACD_patch)
    if save and not tiff: np.save('ACD_patch.npy', ACD_patch)
    if save and tiff:
        write_to_tiff(array = ACD_patch, fn = 'ACD_patch.tiff', 
                      height = 5000, width = 5000, count = 1, 
                      dtype = ACD.dtypes[0], crs = 'EPSG:32650', 
                      transform = ACD.transform)
    return ACD_patch
 
def ACD_geo_bounds():
    ACD = read_tiff('GAO_ACD_30m_unmasked.tif')
    ul_x, ul_y = ACD.xy(6000, 5000, offset='ul')
    lr_x, lr_y = ACD.xy(6000+5000, 5000+5000, offset='ul')
    return ul_x, lr_y, lr_x, ul_y # 421740.0, 490380.0, 571740.0, 640380.0
    
def s2_patch(plot = False, save = False, tiff = True):
    ul_i, lr_j, lr_i, ul_j = ACD_geo_bounds()
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:4326', ul_i, lr_j, lr_i, ul_j)
    
    with rs.open('s2_sabah.tif') as s2 :
        row_start, col_start = s2.index(ul_x, ul_y)
        row_stop, col_stop = row_start + 15000, col_start + 15000
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        s2_window = s2.read([1,2,3,4], window = window)
        s2_window = pre_processing(s2_window)
        assert(s2_window.shape == (4, 15000, 15000))
        
        if plot : plt.imshow(s2.read(1, window = window))
        if save and not tiff : np.save('S2_patch.npy', s2_window)
        if save and tiff : 
            write_to_tiff(array = s2_window, fn = 'S2_patch.tiff', 
                          height = 15000, width = 15000, count = 4, 
                          dtype = 'float64', crs = 'EPSG:4326', transform = s2.transform)   
            
    return s2_window

def l8_patch(plot = False, save = False, tiff = True):
    ul_i, lr_j, lr_i, ul_j = ACD_geo_bounds()
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:4326', ul_i, lr_j, lr_i, ul_j)
    
    with rs.open('l8_sabah.tif') as l8 :
        row_start, col_start = l8.index(ul_x, ul_y)
        row_stop, col_stop = row_start + 5000, col_start + 5000
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        l8_window = l8.read([1,2,3,4], window = window)
        l8_window = pre_processing(l8_window)
        assert(l8_window.shape == (4, 5000, 5000))
        
        if plot : plt.imshow(l8.read(1, window = window))
        if save : np.save('L8_patch.npy', l8_window)
        if save and tiff : 
            write_to_tiff(array = l8_window, fn = 'L8_patch.tiff', 
                          height = 5000, width = 5000, count = 4, 
                          dtype = 'float64', crs = 'EPSG:4326', transform = l8.transform)
        
    return l8_window

def NICFI_patch(bands = [1,2,3,4], plot = False, save = False):
    ul_i, lr_j, lr_i, ul_j = ACD_geo_bounds()
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:3857', ul_i, lr_j, lr_i, ul_j)
    with rs.open('quads/gdal_quads.tiff') as quads :
        row_start, col_start = quads.index(ul_x, ul_y)
        row_stop, col_stop = row_start + 30000, col_start + 30000
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        quads_window = quads.read(bands, window = window)
        quads_window = pre_processing(quads_window)
        assert(quads_window.shape == (len(bands), 30000, 30000))
        if save : 
            f = '_'.join(str(x) for x in bands)
            write_to_tiff(array = quads_window, fn = 'NICFI_patch_bands_{}.tiff'.format(f), 
                          height = 30000, width = 30000, count = len(bands), dtype = 'float64', 
                          crs = 'EPSG:3857', transform = quads.transform)        
        if plot : plt.imshow(quads.read(bands[0], window = window))

    return quads_window



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
