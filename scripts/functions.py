#!/usr/bin/env python
#file navigation tools
from glob import glob
import os
#image analysis library
from skimage import io
#jupyter notebook img display
import stackview
#The fundamental package for scientific computing with Python
import numpy as np
#python image viewer 
import napari
#excel for python
import pandas as pd
import csv
from scipy.ndimage import shift
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot
from cellpose import models
from cellpose.io import imread
from cellpose import core, utils, io, models, metrics
import cv2 as cv
from typing import List
import warnings
warnings.filterwarnings('ignore')

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg

import shapely 
from shapely import Point, Polygon
from shapely import intersection
#from shapely import wkt

import time


# =============================================================================
# This function allows you to ask a yes or no question which either returns a 
# counter 'y' or 'n'. I use this in other functions when deciding to interact 
# with the user. I have usually used this to exit a while True: loop
# =============================================================================
def yes_or_no(question):
    counter = ''
    while counter == '':
        #The user is asked a yes or no question.
        user = input('\n\n!!!\n'+question+'(y/n)\n!!!\n')
        #If they don't either input yes or no in the accepted format the while loop is not broken.
        if user.lower() in ['yes', 'y']:
            counter += 'y'
            #While loop broken and programme run can continue.
            return counter
        elif user.lower() in ['no', 'n']:
            counter+='n'
            #While loop broken and programme run can continue.
            return counter
        else:
            print_info('I\'m sorry I don\'t understand... Please enter (y\\n) in the accepted format')

# =============================================================================
# This function calculates the Laplacian variance of an image. 
# =============================================================================
def lap_variance(img):
    return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))

# =============================================================================
# This function uses the lap_variance function to calulcate the Laplacian
# variance for each slice in a stack and returns which slice (the index) of 
# the slice with the maximum variance.
# =============================================================================
def best_focus_z_slice_id(channel):
    lap_var_z_slice = []
    for img in channel:
        lap_var_z_slice.append(lap_variance(img)) 
    max_var = max(lap_var_z_slice)
    max_var_id = lap_var_z_slice.index(max_var)
    return max_var_id

# =============================================================================
# This function takes the path of a z-stack tif file and splits the stack
# in this case into 4 channels with 41 slices in each.
# =============================================================================
def read_stack(path_array):
    img = io.imread(path_array[0])
    img = np.expand_dims(img,1)
    img = np.reshape(img,(4,41,2304,2304))
    return path_array, img

# =============================================================================
# This function returns the 21 most in focus slices by adding and subtracting 
# 10 from the index of the slice with the maximum variance - a very simple 
# method.
# =============================================================================
def choose_focus_lap(channel_stack):
    focus = [best_focus_z_slice_id(channel_stack)-10, best_focus_z_slice_id(channel_stack)+10]
    #BIGFISH
    #focus = stack.compute_focus(img[i], neighborhood_size=31)
    return focus


# =============================================================================
# This function computes the maximum intesnity projection using the 'most in-fo
# cus' slices. These focussed slices can be input manually or calculated via
# some method.
# =============================================================================
def np_mip(channel_array, focus):
    return np.amax(channel_array[focus[0]:focus[1],...],axis=0,keepdims=False)

# =============================================================================
# This function is a simple wrapper for a bigfish function removing the 
# background of an image - currently unused in the script.
# =============================================================================
def projection_filter(channel_projection):
    return stack.remove_background_gaussian(channel_projection, sigma=3)

# =============================================================================
# This function opens the napari viewer with the Maximum Intensity Projections 
# (MIP) of the channels.
# =============================================================================
def napari_view(files, zproject, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    #viewer= napari.Viewer()
    viewer.add_image(io.imread(files[1]), name='DIC', opacity=1, blending='additive')
    for i in range(zproject.shape[0]):
        viewer.add_image(zproject[i,...],name=channels[i],colormap=colors[i],opacity=1, blending='additive')

# =============================================================================
# This function opens the napari viewer with the MIPs with the corresponding 
# spots for each channel.
# =============================================================================
def napari_view_spots(files, zproject, spots, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    #viewer= napari.Viewer()
    #viewer.add_image(io.imread(files[1]), name='DIC', opacity=1, blending='additive')
    for i in range(len(channels)):
        if channels[i] != 'DAPI':
            coordinates_2d = spots[channels[i]][:, 1:]
            viewer.add_points(coordinates_2d, name=str(channels[i])+' spot', opacity=1, edge_color='red', face_color='transparent')



# =============================================================================
# This function uses the Bigfish detection.detect_spots to compute the coordinates
# of RNA spots for each of the channels that is not the DAPI channel.
# Returns a dictionary of each of the channels and the spot coordinates.
# =============================================================================
def spot_coord(stack, voxel_size=(200, 64.5, 64.5), spot_radius=(200, 70, 70), channels=['CY5', 'CY3', 'CY3.5', 'DAPI']):
    coordinate_dict = {}
    for i in range(len(stack)):
        if channels[i] != 'DAPI':
            spots, threshold = detection.detect_spots(images=stack_example[i], return_threshold=True, 
            voxel_size=voxel_size, spot_radius=spot_radius)
            coordinate_dict[channels[i]] = spots
    return coordinate_dict


# =============================================================================
# This function assigns the dictionary of spot coordinates to each of the cells
# defined by the cell masks.
# =============================================================================
def spot_per_cell(coordinate_dict, first_channel, mask_path, mip_dict):
    coordinate_dict_other = {key: coordinate_dict[key] for key in coordinate_dict if key != first_channel}
    if type(mask_path) == str:
        #cell_label = io.imread(mask_path, plugin='pil')
        cell_label = io.imread(mask_path)
    else:
        cell_label=mask_path
    percell_results = multistack.extract_cell(cell_label=cell_label, 
                        ndim=3, nuc_label=None, rna_coord=coordinate_dict[first_channel], 
                        others_coord=coordinate_dict_other,
                        image=mip_dict[first_channel], 
                        others_image=mip_dict)
    
    return percell_results

# =============================================================================
# This function returns the mask coordinates of the cells in the image.
# =============================================================================
def mask_coordinates(fov_results, dic_shift=[0,0]):
    cell_coords = []
    for i, cell_results in enumerate(fov_results):
        #(min_y, min_x, max_y, max_x)
        y = fov_results[i]['cell_coord'][:, 0] + fov_results[i]['bbox'][0] + dic_shift[0]
        x = fov_results[i]['cell_coord'][:, -1] + fov_results[i]['bbox'][1] + dic_shift[1]
        cell_coords.append(np.dstack((y, x))[0]) 
    return cell_coords


# =============================================================================
# This function converts the mask coordinates to represent their position 
# in the image.
# =============================================================================
def fov_coordinate_translator(i, fov_results, dic_shift=[0,0]):
    y = fov_results[i]['cell_coord'][:, 0] + fov_results[i]['bbox'][0] + dic_shift[0]
    x = fov_results[i]['cell_coord'][:, -1] + fov_results[i]['bbox'][1] + dic_shift[1]
    return np.dstack((y, x))[0]


# =============================================================================
# This function generally converts the coordinates - i.e. also spot coordinates
# to where they are in the image.
# =============================================================================
def fov_RNA_coordinate_translator(i, fov_results, channel):
    y = fov_results[i][channel][:, 1] + fov_results[i]['bbox'][0]
    x = fov_results[i][channel][:, -1] + fov_results[i]['bbox'][1]
    return np.dstack((y, x))[0]

# =============================================================================
# This function computes the centroids of each single cell and the bud and mother
# cell fragments and retuens a dictionary of the index of the mask and the centroids.
# =============================================================================
def mask_centroids(separate_mask_coordinates):
    centroids = {}
    for i in range(len(separate_mask_coordinates)):
        centroids[i] = [float(x) for x in Polygon(separate_mask_coordinates[i]).centroid.wkt[7:-2].split(' ')]
    return centroids

# =============================================================================
# This function uses the centroids for each of the separate masks and if there
# is a mask which contains two centroids those two mask fragments are related.
# These relationships are returned in a 2d-array of related masks.
# =============================================================================
def mother_bud_reunion(whole_mask_coordinates, centroids):
    mother_bud_sep = []
    mother_bud_whole = []
    for i in whole_mask_coordinates:
        counter_dict = {}
        for j in centroids.values():
            counter = shapely.contains_xy(Polygon(i), j[0], j[1])
            if counter == True:
                counter_dict[list(centroids.keys())[list(centroids.values()).index(j)]] = counter
        if len(counter_dict.values()) > 1:
            mother_bud_sep.append(list(counter_dict.keys()))
            mother_bud_whole.append(i)
    return mother_bud_sep

# =============================================================================
# This function calculates the area for each masks in the related pair of masks
# and based on the area each masks is either assigned the label of 'mother' or 
# 'bud'.
# =============================================================================
def mother_or_bud(maskpairs):
    mother_bud_dict = {}
    for i in range(len(maskpairs)):
        area1 = Polygon(fov_sep[maskpairs[i][0]]['cell_coord']).area
        area2 = Polygon(fov_sep[maskpairs[i][1]]['cell_coord']).area
        if area1 > area2:
            mother_bud_dict[maskpairs[i][0]] = 'mother'
            mother_bud_dict[maskpairs[i][1]] = 'bud'
        else:
            mother_bud_dict[maskpairs[i][0]] = 'bud'
            mother_bud_dict[maskpairs[i][1]] = 'mother'
    return mother_bud_dict

# =============================================================================
# This function calculates the summation of the number of spots in each cell 
# for each channel per image.
# =============================================================================
def spots_in_cells(spots, spots_in_cells, channels=['rna_coord', 'CY3', 'CY3.5']):
    spot_num = 0
    for key in list(spots.keys()):
        spot_num += len(spots[key])
    spot_in_cell_num = 0
    for i in spots_in_cells:
        for j in channels:
            spot_in_cell_num += len(i[j])
    proportion = (spot_in_cell_num/spot_num)*100
    return proportion, spot_in_cell_num, spot_num



# =============================================================================
# This function iterates through different shifts and calculates the proportion
# of the mRNA spots that are in cells vs outside of them and returns the max
# proportion along with the corresponding coordinates.
# =============================================================================
def dic_shift_coords(original_fov_whole, original_mask, original_proportion, spot_dict, shift_range=[-15,15]):
    prop_max = original_proportion
    results = []
    for x in range(shift_range[0],shift_range[1],1):
        for y in range(shift_range[0],shift_range[1],1):
            mask_whole_shifted = shift(original_mask, [x, y])
            fov_whole_shifted = spot_per_cell(spot_dict, 'CY5', mask_whole_shifted, projection_dict)
            prop, cells_shifted, total = spots_in_cells(spot_dict, fov_whole_shifted)
            if prop > prop_max:
                prop_max = prop
                results = [prop_max, x, y, fov_whole_shifted]
    return results