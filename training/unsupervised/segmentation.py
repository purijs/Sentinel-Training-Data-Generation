#!/usr/bin/env python
# coding: utf-8

import sys, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from itertools import chain
from training.unsupervised.algorithms.snic import snic, compute_grid
from training.unsupervised.ndim.operations_collections import nd_computations
from training.unsupervised.metric.snic import create_augmented_snic_distance
import rasterio
from skimage import img_as_ubyte,img_as_uint
from skimage import exposure
from osgeo import gdal, ogr
import json, gc

class Segmentation:
    
    '''
    Superpixel based SNIC Implementation for image segmentation and boundary delineation. 
    This is a wrapper for Simple Non-Iterative Clustering algorithm. Use this for segementing different areas of image in unsupervised manner, based
    on spectral differences

    Parameters
    ----------
    rasterio_image_object : rasterio object
        Rasterio object is required for geo-referencing of the output. Just pass in `rasterio.open()` here
        
    seed : int, eg: 400
        Also known as `number of segments` in your image. Larger values give more closed polygons, there is no optimal value. Eg: If you give `100`, 
        it will try to create 100 closed polygons

    output_file_name: string, eg: 'example.geojson'
        Specify the filename/full path where you want to output the geojson file

    Returns :
    ----------
    Function returns:
    1. File Name
    2. GeoJson File
    
    Example
    ----------
    1. Initialize Class
        `obj = Segmentation(rasterio.open('data/myimage.tif'), 100, 'sameple.geojson')`
    
    2. Call this function
        `json, segments = obj.exportGeoJsonFile()`
        
    '''
    
    def __init__(self, rasterio_image_object, seed, output_file_name):
        
        self.image = rasterio_image_object.read()
        self.seed = seed
        self.compactness = 25
        self.image_profile = rasterio_image_object.profile
        self.indices_count = 12
        self.output_file_name = output_file_name
        
        self.image = np.nan_to_num(self.image, copy=True, nan=0.0, posinf=None, neginf=None)
        
    def squeeze_dims(self, img, ndim):
        while img.ndim > ndim:
            img = img[..., 0]
        return img

    def expand_dims(self, img, ndim):
        while img.ndim < ndim:
            img = img[..., np.newaxis]
        return img

    def validate_factor(self, array, factor):
        factor = np.array(factor, dtype=np.int32)
        if np.any(factor <= 0):
            raise ValueError("Factors less than one don't make sense. Factor: {}".format(factor))

        factor = list(factor)
        while len(factor) < len(array.shape):
            factor += [ 1 ]

        return tuple(factor)

    def downsample_with_stride(self, array, factor, num_mips):
        ndim = array.ndim 
        array = self.expand_dims(array, 4)

        factor = self.validate_factor(array, factor)
        if np.all(np.array(factor, int) == 1):
            return []

        results = []
        for mip in range(num_mips):
            array = array[tuple(np.s_[::f] for f in factor)]
            results.append( self.squeeze_dims(array, ndim) )

        return results
    
    def computeFixedIndices(self, image, indices_count):
            
        channels_indices = np.zeros((indices_count, image.shape[1], image.shape[2]))
        
        ### Assume B1 - B12 = 0 - 11
        ### ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11']
        ###   0    1     2   3     4    5    6    7    8     9    10
           
        channels_indices[0] = (image[8] - image[10]) / (image[8] + image[10])
        channels_indices[1] = (image[6] - image[3]) / (image[4] - image[5])
        channels_indices[2] = image[4] + (35 * ((((image[6] + image[3])/2) - image[4])/(image[5] - image[4])))
        channels_indices[3] = (image[4] - image[3]) / (image[4] + image[3])
        channels_indices[4] = (image[5] - image[4]) / (image[4] - image[3])
        channels_indices[5] = ((image[4] - image[3]) - (0.2)*(image[4] - image[2])) * (image[4] - image[3])
        channels_indices[6] = (image[10] - image[6]) / (image[10] + image[6])
        channels_indices[7] = (0.332*(image[2])) + (0.603*(image[3])) + (0.675*(image[5]))+(0.262*(image[8]))
        channels_indices[8] = (image[6] - image[2]) / (image[6] + image[2])
        channels_indices[9] = image[6] / image[3]
        channels_indices[10] = (image[7] - image[3]) / (image[7] + image[3])
        channels_indices[11] = ((0.1 * image[7]) - image[3]) / ((0.1 * image[7]) + image[3])
        
        channels_indices = np.nan_to_num(channels_indices, copy=True, nan=0.0, posinf=None, neginf=None)
        
        temp_a = self.downsample_with_stride(channels_indices[0,:,:], factor=(2,2), num_mips=3)[0]
        temp_b = self.downsample_with_stride(channels_indices[1,:,:], factor=(2,2), num_mips=3)[0]
        temp_c = self.downsample_with_stride(channels_indices[2,:,:], factor=(2,2), num_mips=3)[0]
        temp_d = self.downsample_with_stride(channels_indices[3,:,:], factor=(2,2), num_mips=3)[0]
        temp_e = self.downsample_with_stride(channels_indices[4,:,:], factor=(2,2), num_mips=3)[0]
        temp_f = self.downsample_with_stride(channels_indices[5,:,:], factor=(2,2), num_mips=3)[0]
        temp_g = self.downsample_with_stride(channels_indices[6,:,:], factor=(2,2), num_mips=3)[0]
        temp_h = self.downsample_with_stride(channels_indices[7,:,:], factor=(2,2), num_mips=3)[0]
        temp_i = self.downsample_with_stride(channels_indices[8,:,:], factor=(2,2), num_mips=3)[0]
        temp_j = self.downsample_with_stride(channels_indices[9,:,:], factor=(2,2), num_mips=3)[0]
        temp_k = self.downsample_with_stride(channels_indices[10,:,:], factor=(2,2), num_mips=3)[0]
        temp_l = self.downsample_with_stride(channels_indices[11,:,:], factor=(2,2), num_mips=3)[0]

        data_temp = np.dstack((temp_a, temp_b, temp_c, temp_d, temp_e, temp_f, temp_g, temp_h, temp_i, temp_j, temp_k, temp_l))
        channels_indices = np.transpose(data_temp, (2, 0, 1))
        channels_indices *= 255.0/channels_indices.max()
        
        del data_temp, temp_a, temp_b, temp_c, temp_d, temp_e, temp_f, temp_g, temp_h, temp_i, temp_j, temp_k, temp_l
        gc.collect()
        
        return channels_indices
        
    def reshapeImage(self, input_array, indices_count):
        
        data = self.computeFixedIndices(input_array, indices_count)
        channels = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype=np.uint16)

        for index, image in enumerate([data[i] for i in range(data.shape[0])]):
            stretched=exposure.equalize_hist(image)
            channels[index,:,:] = img_as_uint(stretched)

        return channels
    
    def computeGrid(self):
        
        image = self.reshapeImage(self.image, self.indices_count)
        image_size = image.shape[1:]
        
        grid = compute_grid(image_size, self.seed)
        seeds = list(chain.from_iterable(grid))
        seed_len = len(seeds)
        
        return (grid, seeds, seed_len)
    
    def computeDistance(self):
        
        grid, seeds, seed_length = self.computeGrid()
        
        image = self.reshapeImage(self.image, self.indices_count)
        image_size = image.shape[1:]
        
        distance_metric = create_augmented_snic_distance(image_size, seed_length, self.compactness)
        
        return distance_metric
        
    def runSNIC(self):
        
        original_image = self.reshapeImage(self.image, self.indices_count)
        reshaped_image = np.transpose(original_image, (1, 2, 0)).tolist()
        
        grid, seeds, seed_length = self.computeGrid()
        distance_metric = self.computeDistance()
        compactness = self.compactness
        
        print("Started SNIC")
        segmentation, distances, numSegments = snic(reshaped_image, seed_length, compactness, 
                                                    nd_computations["nd"], distance_metric)
        
        return (segmentation, distances, numSegments)
    
    def exportTiffFile(self):
        
        with rasterio.Env():

            profile = self.image_profile

            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw')

            segmentation, distances, numSegments = self.runSNIC()

            output_tif = 'segmentation_output.tif'
            
            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(np.array(segmentation).astype(rasterio.uint8), 1)
                
        return (output_tif, numSegments)
            
    def exportGeoJsonFile(self):
        
        temp_segmentation_output, numSegments = self.exportTiffFile()
        dst_fieldname=None

        src_ds = gdal.Open( temp_segmentation_output )
        srcband = src_ds.GetRasterBand(1)
        maskband = srcband.GetMaskBand()
        dst_layername = self.output_file_name
        drv = ogr.GetDriverByName('GeoJSON')
        dst_ds = drv.CreateDataSource( dst_layername )
        srs = src_ds.GetSpatialRef()
        dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs = srs )

        if dst_fieldname is None:
            dst_fieldname = 'DN'

        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
        
        gdal.Polygonize( srcband, maskband, dst_layer, dst_field, [], callback=gdal.TermProgress_nocb )

        srcband = None
        src_ds = None
        dst_ds = None
        mask_ds = None
        
        os.remove(temp_segmentation_output)
        return ("Saved GeoJson at:"+ dst_layername)
        