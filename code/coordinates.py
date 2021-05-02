#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:35:34 2021

@author: ghjuliasialelli
"""

import matplotlib.pyplot as plt

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

from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling

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

def s2_patch():
    ul_x, lr_y, lr_x, ul_y = transform_bounds('EPSG:32650', 'EPSG:4326', 421755.0, 490365.0, 571755.0, 640365.0)
    # ul_x, lr_y, lr_x, ul_y = (116.2932643058128, 4.436060081760799, 117.64811844399668, 5.793384413140727)
    with rs.open('Sabah_median_patch1_2016.tif') as s2 :
        row_start, col_start = s2.index(ul_x, ul_y)
        row_stop, col_stop = s2.index(lr_x, lr_y)
        print(row_start, col_start, row_stop, col_stop)
        window = Window.from_slices((row_start, row_stop),(col_start, col_stop))
        s2_window = s2.read([1,2,3],window=window)
    return s2_window

def ACD_patch(fn):
    ACD = read_tiff('GAO_ACD_30m_unmasked.tif')
    ACD_patch = ACD.read()[6000:6000+5000, 5000:5000+5000]
    count, dtype, crs, transform = ACD.count, ACD.dtype[0], ACD.crs, ACD.transform
    write_to_tiff(ACD_patch, fn, count, dtype, crs, transform)
 
    
def ACD_geo_bounds(ACD):
    ul_x, ul_y = ACD.xy(6000, 5000)
    lr_x, lr_y = ACD.xy(6000+5000, 5000+5000)
    return ul_x, lr_y, lr_x, ul_y
    # 421755.0, 490365.0, 571755.0, 640365.0

def read_tiff(file_name):
    return rs.open(r'{}'.format(file_name))

def plot_tiff(file_name):
    img = rs.open(r'{}'.format(file_name))
    show(img)

# en fait, mauvaise idee pcq Windows trop petites
def generate_windows(img, X = 5, Y = 5):
    """
        Returns the Window(left, bottom, right, up) of every X x Y windows in 
        the image, where the ACD is defined. 
        
        -- img : the ACD image
        -- X : width of the windows, default = 5
        -- Y : height of the windows, default = 5
    """
    ll_x, ll_y, ur_x, ur_y = img.bounds
    res = []
    
    left = ll_x
    bottom = ll_y
    right = ll_x + X
    top = ll_y + Y
    
    while right <= ur_x and top <= ur_y : 
        
        left = right
        right = right + X
        
        bottom = top
        top = top + Y

        res.append(Window(left, bottom, right, top))
    
    return res



# can open big Sentinel-2 image if only read window by window: 
# with rs.open('Sabah_median_2016-0000000000-0000032768.tif') as s2 :
#     s2_w1 = s2.read(1, window=Window(0,0,512,256))


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