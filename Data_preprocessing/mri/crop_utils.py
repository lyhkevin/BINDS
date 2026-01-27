from glob import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans
import pandas as pd
from nibabel.orientations import aff2axcodes
import random

def normalize_percentile(volume, lower_percentile=0, upper_percentile=100):
    v_min = np.percentile(volume, lower_percentile)
    v_max = np.percentile(volume, upper_percentile)
    volume = np.clip(volume, v_min, v_max)
    if v_max - v_min != 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)
    return volume

def get_tumor_side(tumor_path):
    tumor_img = nib.load(tumor_path)
    tumor_data = tumor_img.get_fdata()
    tumor_binary = tumor_data > 0
    tumor_binary[:, :, :20] = 0
    tumor_binary[:, :, -20:] = 0
    labeled, num_features = label(tumor_binary)
    if num_features == 0:
        return None
    volumes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_label = np.argmax(volumes) + 1
    com = center_of_mass(tumor_binary, labeled, largest_label)
    x_center = tumor_binary.shape[0] / 2
    if com[0] < x_center:
        return 'L'
    else:
        return 'R'

def crop_tumor_from_side(breast_path, side, crop_size):
    breast_img = nib.load(breast_path)
    affine = breast_img.affine
    axcodes = aff2axcodes(affine)
    breast_data = breast_img.get_fdata()
    breast_mask = breast_data > 0
    coords = np.column_stack(np.nonzero(breast_mask))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
    centers = kmeans.cluster_centers_
    half_crop = (crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2)
    
    if centers[0][0] < centers[1][0]:
        left_label = 0
        right_label = 1
    else:
        left_label = 1
        right_label = 0
        
    left_center = centers[left_label]
    right_center = centers[right_label]
    crop_center = left_center if side == 'L' else right_center
    crop_center = list(crop_center)
    
    labels = kmeans.labels_
    crop_label = left_label if side == 'L' else right_label
    coords_side = coords[labels == crop_label]
    y_max = coords_side[:, 1].max()
    
    crop_center = [int(round(x)) for x in crop_center]
    img_shape = breast_data.shape
    crop_bounds = []
    
    for dim in range(3):
        if dim == 1:
            start = y_max - crop_size[1]
            end = y_max
            if end > img_shape[1]:
                end = img_shape[1]
                start = end - crop_size[1]
        else:
            start = crop_center[dim] - half_crop[dim]
            end = crop_center[dim] + half_crop[dim]
            if start < 0:
                start = 0
                end = crop_size[dim]
            if end > img_shape[dim]:
                end = img_shape[dim]
                start = end - crop_size[dim]
        crop_bounds.append((int(start), int(end)))
    return crop_bounds

def maskcor_extract_3d(mask, padding=(10, 10, 10)):
    p = np.where(mask > 0)
    a = np.zeros([3, 2], dtype=np.int32)
    for i in range(3):
        s = p[i].min()
        e = p[i].max() + 1
        ss = s - padding[i]
        ee = e + padding[i]
        if ss < 0:
            ss = 0
        if ee > mask.shape[i]:
            ee = mask.shape[i]
        a[i, 0] = ss
        a[i, 1] = ee
    return a

def crop_tumor_from_seg(tumor_path, crop_size, center_tumor=False):
    tumor_img = nib.load(tumor_path)
    tumor_data = tumor_img.get_fdata()
    tumor_binary = tumor_data > 0
    tumor_binary[:, :, :20] = 0
    tumor_binary[:, :, -20:] = 0
    
    cor_box = maskcor_extract_3d(gt)
    labeled_array, num_features = label(tumor_binary)
    if num_features == 0:
        return None

    crop_box = np.zeros([3, 2], dtype=np.int32)
    
    if center_tumor:
        center = [(cor_box[i, 0] + cor_box[i, 1]) // 2 for i in range(3)]
        for i in range(3):
            half = crop_size[i] // 2
            start = center[i] - half
            end = start + crop_size[i]
            if start < 0:
                start = 0
                end = crop_size[i]
            if end > gt.shape[i]:
                end = gt.shape[i]
                start = end - crop_size[i]
            crop_box[i] = [start, end]
    else:
        random_x_min = max(cor_box[0, 1] - crop_size[0], 0)
        random_x_max = min(cor_box[0, 0], gt.shape[0] - crop_size[0])
        random_y_min = max(cor_box[1, 1] - crop_size[1], 0)
        random_y_max = min(cor_box[1, 0], gt.shape[1] - crop_size[1])
        random_z_min = max(cor_box[2, 1] - crop_size[2], 0)
        random_z_max = min(cor_box[2, 0], gt.shape[2] - crop_size[2])

        if random_x_min > random_x_max:
            random_x_min, random_x_max = cor_box[0, 0], cor_box[0, 1] - crop_size[0]
        if random_y_min > random_y_max:
            random_y_min, random_y_max = cor_box[1, 0], cor_box[1, 1] - crop_size[1]
        if random_z_min > random_z_max:
            random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]

        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)

        crop_box[0] = [x_random, x_random + crop_size[0]]
        crop_box[1] = [y_random, y_random + crop_size[1]]
        crop_box[2] = [z_random, z_random + crop_size[2]]
        
    return crop_box
