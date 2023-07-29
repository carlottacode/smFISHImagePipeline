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
from skimage.measure import regionprops
from cellpose.io import imread
from cellpose import core, utils, io, models, metrics

import cv2 as cv
from typing import List
import warnings
warnings.filterwarnings('ignore')

import shapely
from shapely import Point, Polygon
from shapely import intersection

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
# This function uses the calulating_laplacian_variance function to calulcate
# the Laplacian variance for each image in a channel image stack and returns
# a list of the Laplacian variance of each slice in a channel in a list.
# =============================================================================
def focus_metric_of_each_z_slice(channel_stack):
    lap_var_of_each_z_slice = []
    # for each image in the stack of one fluorescent channel
    for img in channel_stack:
        # calculate the Laplacian variance and append to a list
        lap_var_of_each_z_slice.append(calulating_laplacian_variance(img))
    return lap_var_of_each_z_slice

# =============================================================================
# This function returns a list of "x" indices of the most in focus slices in a
# channel stack using the top_x_slices function.
# =============================================================================
def choosing_in_focus_slices(channel_stack, number_of_slices):
    # list of the Laplacian focus metric for each image in a channel stack
    focus_list_all = focus_metric_of_each_z_slice(channel_stack)
    # return list of "x" indices of slices with the highest scoring focus metric
    top_in_focus_slices = top_x_slices(focus_list_all, number_of_slices)
    return top_in_focus_slices
# =============================================================================
# This function returns the "x" indices of the most in focus slices
# in terms of a list of image focus scores.
# =============================================================================
def top_x_slices(focus_list_all, x):
    top_in_focus_slices = []
    # until the number of "x" in focus slices has been reached this loop continues
    while len(top_in_focus_slices)<=x:
        # the index of the highest scoring Laplacian focus metric is saved
        max_index = focus_list_all.index(max(focus_list_all))
        # this index is appended to a list
        top_in_focus_slices.append(max_index)
        # this maximum is removed from the original list without changing the
        # indices of the remaning focus scores.
        focus_list_all[max_index] = 0
    # the list of the indices of the top scoring slices is sorted so they are in
    # ascending order
    top_in_focus_slices.sort()
    return top_in_focus_slices

# =============================================================================
# This function computes the maximum intensity projection using the 'most in-fo
# cus' slices. These focussed slices can be input manually or using the
# Laplacian variance.
# =============================================================================
def generating_maximum_projection(channel_array, focus):
    # at each position in the image choose the maximum value of all the slices
    return np.amax(channel_array[focus],axis=0,keepdims=False)

# =============================================================================
# This function is a simple wrapper for a bigfish function removing the
# background of an image - currently unused in the script.
# =============================================================================
def projection_filter(channel_projection):
    return stack.remove_background_gaussian(channel_projection, sigma=3)

# =============================================================================
# This function creates a dictionary of the channels and their corresponding
# image stacks.
# =============================================================================
def split_stack(image, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    image_dictionary = {}
    for i in range(0,len(image)):
        image_dictionary[channels[i]] = (colors[i], image[i])
    return image_dictionary

# =============================================================================
# This function adds to the napari viewer with the Maximum Intensity Projections
# (MIP) of the channels.
# =============================================================================
def napari_view(files, zproject, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    # add the DIC image to the napari viewer
    viewer.add_image(io.imread(files[1]), name='DIC', opacity=0.3, blending='additive')
    # add each MIP using the channel names to identify each one within the viewer
    for i in range(zproject.shape[0]):
        viewer.add_image(zproject[i,...],name=channels[i],colormap=colors[i],opacity=1, blending='additive')

# =============================================================================
# This function adds to the napari viewer with the MIPs with the corresponding
# spots for each channel.
# =============================================================================
def napari_view_spots(files, zproject, spots, channels=['CY5', 'CY3', 'CY3.5', 'DAPI'], colors=['magenta', 'green', 'cyan', 'blue']):
    # for each spot channel (i.e. not DAPI) add the detected spot coordinates
    # to the napari viewer
    for i in range(len(channels)):
        if channels[i] != 'DAPI':
            # use the names of each channel to identify each layer of detected spots
            coordinates_2d = spots[channels[i]][:, 1:]
            viewer.add_points(coordinates_2d, name=str(channels[i])+' spot', opacity=1, edge_color='red', face_color='transparent')

# =============================================================================
# This function adds to the napari viewer with cell masks generated by the
# segmentation model.
# =============================================================================
def napari_view_masks(masks, mask_colors=['pink']):
    # add the masks parsed in as a shape layer
    viewer.add_shapes(masks, shape_type='polygon', edge_width=2, edge_color=colors[0], face_color='transparent')

# =============================================================================
# This function creates a dictionary of the channels and their corresponding
# MIPs. These are named by adding a "P" (for projection) to all channels
# except the first due to specific naming customs of bigfish. Channel names can't
# be repeated.
# =============================================================================
def create_mip_projection_dict(fluor_channels, mip_projection):
    projection_dictionary={}
    # for the index of each channel
    for i in range(len(fluor_channels)):
        # if it is the first channel
        if i == 0:
            # then in the dictionary this can simply be the channel name
            projection_dictionary[str(fluor_channels[i])] = mip_projection[i]
        else:
            # but if this is a subsequent channel this name will already appear
            # in a bigfish dictionary of results and therefore a "P" is added
            projection_dictionary[fluor_channels[i]+'P'] = mip_projection[i]
    return projection_dictionary

# =============================================================================
# This function uses the bigfish detection.detect_spots to compute the coordinates
# of RNA spots for each of the channels omitting the DAPI channel.
# Returns a dictionary of each of the channels and the spot coordinates.
# =============================================================================
def detecting_mRNA_spots(stack, voxel_size=(200, 64.5, 64.5), spot_radius=(200, 70, 70), channels=['CY5', 'CY3', 'CY3.5', 'DAPI']):
    coordinate_dict = {}
    for i in range(len(stack)):
        if channels[i] != 'DAPI':
            # if the channel isn't DAPI detect spots by detection and decomposition
            spots, threshold = detection.detect_spots(images=stack[i], return_threshold=True,
            voxel_size=voxel_size, spot_radius=spot_radius)
            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(image=stack[i],
            spots=spots, voxel_size=voxel_size, spot_radius=spot_radius,
                    alpha=0.7,  # alpha impacts the number of spots per candidate region
                    beta=1,  # beta impacts the number of candidate regions to decompose
                    gamma=5)  # gamma the filtering step to denoise the image
            # add spot coordinates to a dictionary of spots where the key is the channel name
            coordinate_dict[channels[i]] = spots_post_decomposition
    return coordinate_dict

# =============================================================================
# This function assigns the dictionary of spot coordinates to each of the cells
# defined by the cell masks.
# =============================================================================
def extracting_spots_per_cell(coordinate_dict, mask_path, mip_dict):
    # formatting the channel names for bigfish function
    first_channel = list(coordinate_dict.keys())[0]
    coordinate_dict_other = {key: coordinate_dict[key] for key in coordinate_dict if key != first_channel}
    # if mask is a file path (i.e. segmentation has happened outside of the pipeline)
    # the mask file is read
    if type(mask_path) == str:
        cell_label = io.imread(mask_path)
    # if the masks have been generated via the pipeline they are simply assigned
    else:
        cell_label=mask_path
    # bigfish function is used to extract the mRNA spots per cell
    percell_results = multistack.extract_cell(cell_label=cell_label,
                        ndim=3, nuc_label=None, rna_coord=coordinate_dict[first_channel],
                        others_coord=coordinate_dict_other,
                        image=mip_dict[first_channel],
                        others_image=mip_dict)
    # the data is returned as a dictionary
    return percell_results

# =============================================================================
# This function returns the mask coordinates so that they can be mapped onto
# the DIC and the MIPs. If two Cellpose models are used to identify mother and
# bud cells this function is used to map cell masks from the two different
# models onto each other.
# =============================================================================
def mapping_mask_coordinates(shifted_fov_results):
    cell_coords = []
    # for each cell in extracted cell dictionary
    for i, cell_results in enumerate(shifted_fov_results):
        #(min_y, min_x, max_y, max_x)
        # to each y coordinate in the "cell_coord" array add on the minimum y value stored a "bbox" tuple
        y = shifted_fov_results[i]['cell_coord'][:, 0] + shifted_fov_results[i]['bbox'][0]
        # to each x coordinate in the "cell_coord" array add on the minimum x value stored a "bbox" tuple
        x = shifted_fov_results[i]['cell_coord'][:, -1] + shifted_fov_results[i]['bbox'][1]
        # add all the cell arrays to a list
        cell_coords.append(np.dstack((y, x))[0])
    # return list of mapped cell masks
    return cell_coords

# =============================================================================
# This function computes the centroids of each segemented cell from the
# separate Cellpose model and returns a dictionary of the index of the mask
# and the centroids.
# =============================================================================
def mask_centroids(separate_mask_coordinates):
    centroids = {}
    # using index as the cell ID anf key in the centroids dictionary
    for i in range(len(separate_mask_coordinates)):
        # use Shapely functionality to calculate the centroids of each cell mask shape
        centroids[i] = [float(x) for x in Polygon(separate_mask_coordinates[i]).centroid.wkt[7:-2].split(' ')]
    return centroids

# =============================================================================
# This function uses the centroids for each of the separate masks and isolates
# masks from the "whole" Cellpose model containing two centroids
# as those two mask fragments are related. These relationships are returned
# in a 2d-array of related masks.
# =============================================================================
def mother_bud_reunion(whole_mask_coordinates, centroids):
    mother_bud_sep = []
    mother_bud_whole = []
    # for each mask generated from the "whole" Cellpose model
    for i in whole_mask_coordinates:
        # initiate a counter
        counter_dict = {}
        # for the centroids in the centroid dictionary
        for j in centroids.values():
            counter = shapely.contains_xy(Polygon(i), j[0], j[1])
            # if the centroid is contained within the mask generated from the "whole" Cellpose model
            # it is added to the counter dictionary - with the ID of the "separate" mask as a key
            if counter == True:
                counter_dict[list(centroids.keys())[list(centroids.values()).index(j)]] = counter
        # if the counter dictionary has more than 1 value
        if len(counter_dict.values()) > 1:
            # then the ID of the "separate" mask is appended to a list
            mother_bud_sep.append(list(counter_dict.keys()))
            mother_bud_whole.append(i)
    # this list is returned after all masks generated from the "whole" Cellpose model
    # have been iterated through
    return mother_bud_sep

# =============================================================================
# This function calculates the area for each mask in the related pair of masks
# and based on the area - each masks is either assigned the label of 'mother' or
# 'bud' (with smaller area being 'bud' and larger area 'mother').
# =============================================================================
def mother_or_bud(maskpairs):
    mother_bud_dict = {}
    for i in range(len(maskpairs)):
        # calculate the area of each related mask
        area1 = Polygon(fov_sep[maskpairs[i][0]]['cell_coord']).area
        area2 = Polygon(fov_sep[maskpairs[i][1]]['cell_coord']).area
        # if area1 is bigger than area2
        if area1 > area2:
            # then the mask from which area1 was calculated from is assigned 'mother'
            mother_bud_dict[maskpairs[i][0]] = ('mother', 'corresponding bud: '+str(maskpairs[i][1]))
            mother_bud_dict[maskpairs[i][1]] = ('bud', 'corresponding mother: '+str(maskpairs[i][0]))
        else:
            # else the mask from which area2 was calculated is assigned 'mother'
            mother_bud_dict[maskpairs[i][0]] = ('bud', 'corresponding mother: '+str(maskpairs[i][1]))
            mother_bud_dict[maskpairs[i][1]] = ('mother', 'corresponding bud: '+str(maskpairs[i][0]))
    # the ID of the corresponding cell fragment is also added to a dictionary which is returned
    return mother_bud_dict

# =============================================================================
# This function calculates the total proportion of the number of spots in each cell
# vs the number of spots outside of cell boundaries for each channel per image.
# =============================================================================
def counting_spots_in_cells(spots, spots_in_cells, channels=['rna_coord', 'CY3', 'CY3.5']):
    # calculate the total number of spots detected
    spot_num = 0
    for key in list(spots.keys()):
        spot_num += len(spots[key])

    # calculate the total number of spots assigned to cells
    spot_in_cell_num = 0
    for i in spots_in_cells:
        for j in channels:
            spot_in_cell_num += len(i[j])

    # use these two figures to calculate the proportion of sopts in cells
    proportion = (spot_in_cell_num/spot_num)*100
    return proportion, spot_in_cell_num, spot_num

# =============================================================================
# BRUTE FORCE SHIFT CORRECTION APPROACH - currently unused in script
# This function iterates through different shifts and calculates the proportion
# of the mRNA spots that are in cells vs outside of them and returns the max
# proportion along with the corresponding coordinates.
# =============================================================================
def dic_shift_coords(original_fov_whole, original_mask, original_proportion, spot_dict, shift_range=[-15,15]):
    prop_max = original_proportion
    results = []
    for x in range(shift_range[0],shift_range[1],1):
        for y in range(shift_range[0],shift_range[1],1):
            # for each combination of x and y coordinate shift the masks
            mask_whole_shifted = shift(original_mask, [x, y])
            # calculate the new proportion of spots in cells
            fov_whole_shifted = extracting_spots_per_cell(spot_dict, mask_whole_shifted, projection_dict)
            prop, cells_shifted, total = counting_spots_in_cells(spot_dict, fov_whole_shifted)
            # if the porportion is higher than the current maximum save this along with the
            # coordinates that have generated this maximum
            if prop > prop_max:
                prop_max = prop
                results = [prop_max, x, y, fov_whole_shifted]
    # after all coordinate combinations have been iterated through the maximum proportion
    # and corresponding coordinates are returned
    return results

# =============================================================================
# This function shifts the masks and calculates the new proportion of spots
# within cells with this shift and the new proportion is returned.
# =============================================================================
def shifting_and_counting(original_mask, spot_dict, x, y, projection_dict):
    # shift the mask
    mask_whole_shifted = shift(original_mask, [x, y])
    # use the shifted mask to extract spots in cells
    fov_whole_shifted = extracting_spots_per_cell(spot_dict, mask_whole_shifted, projection_dict)
    # use the new number of spots in cells to calculate the new proportion
    prop, cells_shifted, total = counting_spots_in_cells(spot_dict, fov_whole_shifted)
    return prop

# =============================================================================
# This function avoids the calulcation of proportion values for coordinate
# combinations which have already been calculated.
# =============================================================================
def checking_shift_dictionary(shift_coord, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    # if the coordinates that need to be checked are already in the proportion dictionary
    if str(shift_coord) in proportion_dictionary:
        # the proportion is simply the previously calculated proportion value in the dictionary
        proportion = proportion_dictionary[str(shift_coord)]
    else:
        # otherwise the shifting_and_counting function is used to calculate the new proportion
        proportion = shifting_and_counting(original_mask, spot_dict, shift_coord[0], shift_coord[1], projection_dictionary)
        proportion_dictionary[str(shift_coord)] = proportion
    # this proportion is returned
    return proportion

# =============================================================================
# This function defines all the neighbours around a point and returns the
# neighbours as a list.
# =============================================================================
def defining_neighbours(x, y, proportion_dictionary, spot_dict, original_mask):
    # centre coordinate
    start = [x, y]
    # neighbours
    up_shift = [x, y+1]
    upleft_shift = [x-1, y+1]
    upright_shift = [x+1, y+1]
    down_shift = [x, y-1]
    downleft_shift = [x-1, y-1]
    downright_shift = [x+1, y-1]
    left_shift = [x-1, y]
    right_shift = [x+1, y]
    # all the neighbours are placed in a list
    shift_neighbours = [start, up_shift, upleft_shift, upright_shift,down_shift, downleft_shift, downright_shift,left_shift, right_shift]
    # this list is returned
    return shift_neighbours

# =============================================================================
# This function calculates the proportions of spots in cells when the masks are
# shifted around the centre i.e. all the proportions of neighbour shifts
# are calculated. Calculating the same proportion twice is prevented by
# checking the proportion dictionary.
# =============================================================================
def calculating_neighbour_proportions(prev_max_prop, shift_neightbours_list, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    prop_max = [prev_max_prop, [0, 0]]
    # for each of the neighbours in the list
    for i in shift_neightbours_list:
        # check the dictionary or calculate the proportion for each neighbour shift
        prop = checking_shift_dictionary(i, proportion_dictionary, spot_dict, original_mask, projection_dictionary)
        # if the proportion calculated is higher than the previous maximum proportion
        if prop > prop_max[0]:
            # this value along with the corresponding coordinates are saved
            prop_max = [prop, i]
    # when all neighbours have been checked the maximum proportion and coordinates of these neighbours are returned
    return prop_max

# =============================================================================
# This function finds the shift with the maximum proportion of spots within
# cells.
# =============================================================================
def finding_max_proportion_of_image(prev_max_prop, x, y, proportion_dictionary, spot_dict, original_mask, projection_dictionary):
    # neighbours are defined for the starting point
    neighbour_list = defining_neighbours(x, y, proportion_dictionary, spot_dict, original_mask)
    # proportions for all the neighbouring shifts are calculated
    maximum_neighbour_proportion = calculating_neighbour_proportions(prev_max_prop, neighbour_list, proportion_dictionary, spot_dict, original_mask, projection_dictionary)
    # if the maximum proportion of the neighbours is larger than the previous maximum
    if maximum_neighbour_proportion[0] > prev_max_prop:
        # print the current shift information and rerun this function using this maximum and the coordinates that generated them as the starting point
        print(f'Current maximum proportion {maximum_neighbour_proportion[0]:.2f} and the cooresponding coordinates: {maximum_neighbour_proportion[1]}')
        return finding_max_proportion_of_image(maximum_neighbour_proportion[0], maximum_neighbour_proportion[1][0], maximum_neighbour_proportion[1][1], proportion_dictionary, spot_dict, original_mask, projection_dictionary)
    else:
        # otherwise, if none of the neighbours are bigger, the maximum of the image has been found and the proportion and the coordinates are returned
        return [prev_max_prop, [x, y]]

# =============================================================================
# This function calculates mRNA localisation metrics of each cell using
# adapted bigfish functions.
# =============================================================================
def calculating_bigfish_metrics_per_channel(cell_mask, rna_coord, smfish, ndim, channel, voxel_size_yx=103, foci_coord=None):
    # calculate distance metrics
    distance, distance_names = features_distance(rna_coord, _get_distance_cell(cell_mask), cell_mask, ndim, channel)
    # calculate protrusion metrics
    protrusion, protrusion_names = features_protrusion(rna_coord, cell_mask, ndim, voxel_size_yx, channel)
    # calculate dispersion metrics
    dispersion, dispersion_names = features_dispersion(smfish, rna_coord, _get_centroid_rna(rna_coord, 2), cell_mask, _get_centroid_surface(cell_mask), ndim, channel, False)
    #features_foci(rna_coord, foci_coord, ndim)
    # combine all metrics into a numpy array
    features = np.concatenate((distance, protrusion, dispersion), axis=0)
    feature_names = distance_names + protrusion_names + dispersion_names
    # return the array of features
    return np.column_stack((features, feature_names))

# =============================================================================
# This function calculates the area of each cell using a bigfish function.
# =============================================================================
def calculating_bigfish_area(cell_mask):
    # area is calculated
    area, area_names = features_area(cell_mask)
    # area features returned as an array
    return [area[0], area_names[0]]

# =============================================================================
# This function is a bigfish function which is required for the adapted bigfish
# functions to run. It caluclates the centroid coordinates of a 2-d binary surface.
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
# This function uses the dictionary containing the mask IDs and their label
# (i.e. whether they are a mother or a bud) to add this information to the array
# of calculated feature metrics.
# =============================================================================
def assigning_cell_type_DataFrame(cell_id, celltype_dictionary):
    # if the ID of the cell is in the dictionary the value of the dictionary is
    # returned (either 'mother' or 'bud')
    if cell_id in celltype_dictionary.keys():
        return [celltype_dictionary[cell_id], 'cell_type']
    else:
        # otherwise the cell is not a 'mother' or 'bud' segment and is therefore
        # denoted as a 'single cell'
        return ['single cell', 'cell_type']

# =============================================================================
# This function uses the functions calculating_bigfish_metrics_per_channel,
# calculating_bigfish_area and assigning_cell_type_DataFrame to calculate
# the metrics for each channel in the image and return the information in a DataFrame.
# =============================================================================
def writing_metrics_to_DataFrame(fov, channels, im_channels, celltype_dictionary):
    one_cell = []
    listDict = []
    # for each cell
    for i in range(len(fov)):
        # for each channel
        for j in range(len(channels)):
            # calculate the bigfish metrics
            one_cell.append(calculating_bigfish_metrics_per_channel(fov[i]["cell_mask"], fov[i][channels[j]], fov[i][im_channels[j]], 3, channels[j]))
            # count the number of mRNA molecules
            one_cell.append([[len(fov[i][channels[j]]), 'number_of_mRNA_'+channels[j]]])
        # calulate the area for each cell
        one_cell.append([calculating_bigfish_area(fov[i]["cell_mask"])])
        #print(calculating_bigfish_area(fov[i]["cell_mask"]))
        # if a celltype_dictionary has been created
        if celltype_dictionary != None:
            # then celltypes are added to the list of cell information
            one_cell.append([assigning_cell_type_DataFrame(i, celltype_dictionary)])
        # create a dictionary from the array which contains the metrics and the name of the metric calculated
        # append each cell dictionary in a list
        listDict.append(dict(zip(np.concatenate(one_cell)[:,-1],np.concatenate(one_cell)[:,0])))
    # use the list of dictionaries containing the metrics of each cell to create a DataFrame
    return pd.DataFrame(listDict)

# =============================================================================
# This function saves the DataFrame created in an excel document to the output path
# specified by the user. If the "separate" Cellpose model has been used as well
# as the "whole" cell model both of these DataFrames are output on different sheets.
# =============================================================================
def writing_dataframe_to_excel(output_path, whole_df_output, sep_df_output):
    with pd.ExcelWriter(results_excel_path, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_numbers': True}}) as writer:
        whole_df_output.to_excel(writer, sheet_name='whole_cells')
        # If there is a separate DataFrame this is output on a different sheet
        if sep_df_output != None:
            sep_df_output.to_excel(writer, sheet_name='mother_buds_separate')
