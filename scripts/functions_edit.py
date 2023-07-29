#!/usr/bin/env python
#file navigation tools
from glob import glob
import os
#image analysis library
from skimage import io
#jupyter notebook img display
#import stackview
#The fundamental package for scientific computing with Python
import numpy as np
#python image viewer
import napari
#excel for python
import pandas as pd

import csv
#file navigation tools
from glob import glob
import time
from scipy.ndimage import shift

import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot
import bigfish.classification as classification
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
# This function takes the path of a z-stack tif file and splits the stack
# in this case into 4 channels with 41 slices in each.
# =============================================================================
def read_stack(path_array, number_of_slices_per_channel, channels):
    img = io.imread(path_array[0])
    img = np.expand_dims(img,1)
    img = np.reshape(img,(len(channels),number_of_slices_per_channel,2304,2304))
    return path_array, img

# =============================================================================
# This function calculates the Laplacian variance of an image.
# =============================================================================
def calulating_laplacian_variance(img):
    return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))

# =============================================================================
# This function calculates the Laplacian variance of an image.
# =============================================================================
def calculating_sobel_gradient_magnitude(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.magnitude(sobel_x, sobel_y)

# =============================================================================
# This function uses the lap_variance function to calulcate the Laplacian
# variance for each slice in a stack and returns which slice (the index) of
# the slice with the maximum variance.
# =============================================================================
def focus_metric_of_each_z_slice(channel_stack):
    lap_var_of_each_z_slice = []
    for img in channel_stack:
        lap_var_of_each_z_slice.append(calulating_laplacian_variance(img))
    return lap_var_of_each_z_slice

# =============================================================================
# This function returns the 21 most in focus slices by adding and subtracting
# 10 from the index of the slice with the maximum variance - a very simple
# method.
# =============================================================================
def choosing_in_focus_slices(channel_stack, number_of_slices):
    focus_list_all = focus_metric_of_each_z_slice(channel_stack)
    top_in_focus_slices = top_x_slices(focus_list_all, number_of_slices)
    return top_in_focus_slices
# =============================================================================
# This function returns the x most in focus slices in terms of their laplacian
# variance.
# =============================================================================
def top_x_slices(focus_list_all, x):
    top_in_focus_slices = []
    while len(top_in_focus_slices)<=x:
        max_index = focus_list_all.index(max(focus_list_all))
        top_in_focus_slices.append(max_index)
        focus_list_all[max_index] = 0
    top_in_focus_slices.sort()
    return top_in_focus_slices

# =============================================================================
# This function computes the maximum intesnity projection using the 'most in-fo
# cus' slices. These focussed slices can be input manually or calculated via
# some method.
# =============================================================================
def generating_maximum_projection(channel_array, focus):
    return np.amax(channel_array[focus],axis=0,keepdims=False)

def np_mip_og(channel_array, focus):
    return np.amax(channel_array[focus[0]:focus[1],...],axis=0,keepdims=False)

# =============================================================================
# This function is a simple wrapper for a bigfish function removing the
# background of an image - currently unused in the script.
# =============================================================================
def projection_filter(channel_projection):
    return stack.remove_background_gaussian(channel_projection, sigma=3)

# =============================================================================
#
#
# =============================================================================
def split_stack(image, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    image_dictionary = {}
    for i in range(0,len(image)):
        image_dictionary[channels[i]] = (colors[i], image[i])
    return image_dictionary
# =============================================================================
# This function opens the napari viewer with the Maximum Intensity Projections
# (MIP) of the channels.
# =============================================================================
def napari_view(files, zproject, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    #viewer= napari.Viewer()
    viewer.add_image(io.imread(files[1]), name='DIC', opacity=0.3, blending='additive')
    for i in range(zproject.shape[0]):
        viewer.add_image(zproject[i,...],name=channels[i],colormap=colors[i],opacity=1, blending='additive')

# =============================================================================
# This function opens the napari viewer with the MIPs with the corresponding
# spots for each channel.
# =============================================================================
def napari_view_spots(files, zproject, spots, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    for i in range(len(channels)):
        if channels[i] != 'DAPI':
            coordinates_2d = spots[channels[i]][:, 1:]
            viewer.add_points(coordinates_2d, name=str(channels[i])+' spot', opacity=1, edge_color='red', face_color='transparent')

# =============================================================================
#
#
# =============================================================================
def napari_view_masks(mask_sep,mask_whole, mask_colors=['pink', 'lightblue']):
    viewer.add_shapes(mask_sep, shape_type='polygon', edge_width=2, edge_color=colors[0], face_color='transparent')
    viewer.add_shapes(mask_whole, shape_type='polygon', edge_width=2, edge_color=colors[1], face_color='transparent')

# =============================================================================
#
#
# =============================================================================
def create_mip_projection_dict(fluor_channels, mip_projection):
    projection_dictionary={}
    for i in range(len(fluor_channels)):
        if i == 0:
            projection_dictionary[str(fluor_channels[i])] = mip_projection[i]
        else:
            projection_dictionary[fluor_channels[i]+'P'] = mip_projection[i]
    return projection_dictionary

# =============================================================================
# This function uses the Bigfish detection.detect_spots to compute the coordinates
# of RNA spots for each of the channels that is not the DAPI channel.
# Returns a dictionary of each of the channels and the spot coordinates.
# =============================================================================
def detecting_mRNA_spots(stack, voxel_size=(200, 64.5, 64.5), spot_radius=(200, 70, 70), channels=['CY5', 'CY3', 'CY3.5', 'DAPI']):
    coordinate_dict = {}
    for i in range(len(stack)):
        if channels[i] != 'DAPI':
            spots, threshold = detection.detect_spots(images=stack[i], return_threshold=True,
            voxel_size=voxel_size, spot_radius=spot_radius)
            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(image=stack[i],
            spots=spots, voxel_size=voxel_size, spot_radius=spot_radius,
                    alpha=0.7,  # alpha impacts the number of spots per candidate region
                    beta=1,  # beta impacts the number of candidate regions to decompose
                    gamma=5)  # gamma the filtering step to denoise the image
            coordinate_dict[channels[i]] = spots_post_decomposition
    return coordinate_dict


# =============================================================================
# This function assigns the dictionary of spot coordinates to each of the cells
# defined by the cell masks.
# =============================================================================
def extracting_spots_per_cell(coordinate_dict, mask_path, mip_dict):
    first_channel = list(coordinate_dict.keys())[0]
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
def mapping_mask_coordinates(shifted_fov_results):
    cell_coords = []
    for i, cell_results in enumerate(shifted_fov_results):
        #(min_y, min_x, max_y, max_x)
        y = shifted_fov_results[i]['cell_coord'][:, 0] + shifted_fov_results[i]['bbox'][0]
        x = shifted_fov_results[i]['cell_coord'][:, -1] + shifted_fov_results[i]['bbox'][1]
        cell_coords.append(np.dstack((y, x))[0])
    return cell_coords



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
            mother_bud_dict[maskpairs[i][0]] = ('mother', 'corresponding bud: '+str(maskpairs[i][1]))
            mother_bud_dict[maskpairs[i][1]] = ('bud', 'corresponding mother: '+str(maskpairs[i][0]))
        else:
            mother_bud_dict[maskpairs[i][0]] = ('bud', 'corresponding mother: '+str(maskpairs[i][1]))
            mother_bud_dict[maskpairs[i][1]] = ('mother', 'corresponding bud: '+str(maskpairs[i][0]))
    return mother_bud_dict

# =============================================================================
# This function calculates the summation of the number of spots in each cell
# for each channel per image.
# =============================================================================
def counting_spots_in_cells(spots, spots_in_cells, channels=['rna_coord', 'CY3', 'CY3.5']):
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
            fov_whole_shifted = extracting_spots_per_cell(spot_dict, mask_whole_shifted, projection_dict)
            prop, cells_shifted, total = counting_spots_in_cells(spot_dict, fov_whole_shifted)
            if prop > prop_max:
                prop_max = prop
                results = [prop_max, x, y, fov_whole_shifted]
    return results

# =============================================================================
# This function
#
# =============================================================================
def shifting_and_counting(original_mask, spot_dict, x, y, projection_dict):
    mask_whole_shifted = shift(original_mask, [x, y])
    fov_whole_shifted = extracting_spots_per_cell(spot_dict, mask_whole_shifted, projection_dict)
    prop, cells_shifted, total = counting_spots_in_cells(spot_dict, fov_whole_shifted)
    return prop

# =============================================================================
# This function
#
# =============================================================================
def checking_shift_dictionary(shift_coord, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    if str(shift_coord) in proportion_dictionary:
        proportion = proportion_dictionary[str(shift_coord)]
    else:
        proportion = shifting_and_counting(original_mask, spot_dict, shift_coord[0], shift_coord[1], projection_dictionary)
        proportion_dictionary[str(shift_coord)] = proportion
    return proportion

# =============================================================================
# This function
#
# =============================================================================
def defining_neighbours(x, y, proportion_dictionary, spot_dict, original_mask):
    start = [x, y]
    up_shift = [x, y+1]
    upleft_shift = [x-1, y+1]
    upright_shift = [x+1, y+1]
    down_shift = [x, y-1]
    downleft_shift = [x-1, y-1]
    downright_shift = [x+1, y-1]
    left_shift = [x-1, y]
    right_shift = [x+1, y]
    shift_neighbours = [start, up_shift, upleft_shift, upright_shift,down_shift, downleft_shift, downright_shift,left_shift, right_shift]
    return shift_neighbours

# =============================================================================
# This function
#
# =============================================================================
def calculating_neighbour_proportions(prev_max_prop, shift_neightbours_list, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    prop_max = [prev_max_prop, [0, 0]]
    for i in shift_neightbours_list:
        prop = checking_shift_dictionary(i, proportion_dictionary, spot_dict, original_mask, projection_dictionary)
        if prop > prop_max[0]:
            prop_max = [prop, i]
    return prop_max

# =============================================================================
# This function
#
# =============================================================================
def finding_max_proportion_of_image(prev_max_prop, x, y, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    neighbour_list = defining_neighbours(x, y, proportion_dictionary, spot_dict, original_mask)
    maximum_neighbour_proportion = calculating_neighbour_proportions(prev_max_prop, neighbour_list, proportion_dictionary, spot_dict, original_mask, projection_dictionary)
    if maximum_neighbour_proportion[0] > prev_max_prop:
        print(f'Current maximum proportion {maximum_neighbour_proportion[0]:.2f} and the cooresponding coordinates: {maximum_neighbour_proportion[1]}')
        return finding_max_proportion_of_image(maximum_neighbour_proportion[0], maximum_neighbour_proportion[1][0], maximum_neighbour_proportion[1][1], proportion_dictionary, spot_dict, original_mask, projection_dictionary)
    else:
        return [prev_max_prop, [x, y]]

# =============================================================================
# This function
#
# =============================================================================

def calculating_bigfish_metrics_per_channel(cell_mask, rna_coord, smfish, ndim, channel, voxel_size_yx=103, foci_coord=None):
    distance, distance_names = features_distance(rna_coord, _get_distance_cell(cell_mask), cell_mask, ndim, channel)
    protrusion, protrusion_names = features_protrusion(rna_coord, cell_mask, ndim, voxel_size_yx, channel)
    dispersion, dispersion_names = features_dispersion(smfish, rna_coord, _get_centroid_rna(rna_coord, 2), cell_mask, _get_centroid_surface(cell_mask), ndim, channel, False)
    #features_foci(rna_coord, foci_coord, ndim)
    features = np.concatenate((distance, protrusion, dispersion), axis=0)
    feature_names = distance_names + protrusion_names + dispersion_names
    return np.column_stack((features, feature_names))
# =============================================================================
# This function
#
# =============================================================================

def calculating_bigfish_area(cell_mask):
    area, area_names = features_area(cell_mask)
    return [area[0], area_names[0]]

from skimage.measure import regionprops
# =============================================================================
# This function
#
# =============================================================================
def _get_centroid_surface(mask):
    """Get centroid coordinates of a 2-d binary surface.

    Parameters
    ----------
    mask : np.ndarray, bool
        Binary surface with shape (y, x).

    Returns
    -------
    centroid : np.ndarray, np.int
        Coordinates of the centroid with shape (2,).

    """
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid

# =============================================================================
# This function
#
# =============================================================================
def assigning_cell_type_DataFrame(cell_id, celltype_dictionary):
    if cell_id in celltype_dictionary.keys():
        return [celltype_dictionary[cell_id], 'cell_type']
    else:
        return ['single cell', 'cell_type']
# =============================================================================
# This function
#
# =============================================================================
def writing_metrics_to_DataFrame(fov, channels, im_channels, celltype_dictionary):
    one_cell = []
    listDict = []
    for i in range(len(fov)):
        for j in range(len(channels)):
            one_cell.append(calculating_bigfish_metrics_per_channel(fov[i]["cell_mask"], fov[i][channels[j]], fov[i][im_channels[j]], 3, channels[j]))
            one_cell.append([[len(fov[i][channels[j]]), 'number_of_mRNA_'+channels[j]]])
        one_cell.append([calculating_bigfish_area(fov[i]["cell_mask"])])
        #print(calculating_bigfish_area(fov[i]["cell_mask"]))
        one_cell.append([assigning_cell_type_DataFrame(i, celltype_dictionary)])
        listDict.append(dict(zip(np.concatenate(one_cell)[:,-1],np.concatenate(one_cell)[:,0])))
    return pd.DataFrame(listDict)

# =============================================================================
# This function
#
# =============================================================================
def writing_dataframe_to_excel(output_path, whole_df_output, sep_df_output):
    with pd.ExcelWriter(results_excel_path, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_numbers': True}}) as writer:
        whole_df_output.to_excel(writer, sheet_name='whole_cells')
        sep_df_output.to_excel(writer, sheet_name='mother_buds_separate')
