import plistlib
import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
import os
import cv2
from PIL import Image

def load_dicom_image(dicom_path):
    dcm = pydicom.dcmread(dicom_path, force=True)
    if hasattr(dcm, 'WindowCenter'):
        if type(dcm.WindowCenter) == pydicom.multival.MultiValue:
            img = histogramTransfer(dcm)
        else:
            window_center = float(dcm.WindowCenter)
            window_width = float(dcm.WindowWidth)
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            img = np.clip(dcm.pixel_array, window_min, window_max)
            img = ((img - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    else:
        img = (dcm.pixel_array / dcm.pixel_array.max() * 255).astype('uint8')
    img = img.astype(np.uint8)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 255 - img
    return img
def get_metatdata(dicom_path):
    dcm = pydicom.dcmread(dicom_path, force=True)
    if (0x0008, 0x0068) in dcm:
        presentation = dcm[(0x0008, 0x0068)].value
        if presentation != 'FOR PRESENTATION':
            return False
    series_description, clinical_view = None, None
    if (0x0008, 0x103e) in dcm:
        series_description = dcm[(0x0008, 0x103e)].value
    if (0x0018, 0x1030) in dcm:
        clinical_view = dcm[(0x0018, 0x1030)].value
    print(series_description, clinical_view)
    if series_description == None and clinical_view == None:
        return False
    series_description = '' if series_description is None else series_description
    clinical_view = '' if clinical_view is None else clinical_view
    view = series_description + ' ' + clinical_view
    return view
    # view_mapping = {
    #         ('L CC', 'LCC'): 'L CC',
    #         ('L MLO', 'LMLO'): 'L MLO', 
    #         ('R CC', 'RCC'): 'R CC',
    #         ('R MLO', 'RMLO'): 'R MLO'
    #     }
    # view = ''
    # for keywords, view_type in view_mapping.items():
    #     if any(keyword in series_description for keyword in keywords):
    #         view = view_type
    # return view

def histogramTransfer(dcm):
    data = np.array(dcm.pixel_array, dtype=np.int16)
    bin_size = 10
    hist_range = data.max() - data.min()
    hist, bin_edges = np.histogram(data.astype(np.uint), bins=int(hist_range / bin_size))
    hist_diff = np.diff(hist)
    start_i = int(500 / bin_size)
    hist_min = int(hist[start_i])
    hist_max = int(hist[-1])
    footlength = int(hist_range / 5)
    for i in range(start_i, len(hist) - int(footlength / bin_size)):
        if hist[i] > 50 * bin_size and np.median(
                hist[i:i + int(footlength / bin_size)]) > 100 * bin_size and np.median(
            hist_diff[i - 1:i + 1]) > 10:
            hist_min = int(bin_edges[i])
            break
        footlength = int(hist_range / 10)
        for i in range(len(hist) - 1, start_i + int(footlength / bin_size), -1):
            if np.median(hist[i - int(footlength / bin_size):i]) > 30 * bin_size and hist_diff[i - 1] < 0:
                hist_max = int(bin_edges[i])
                break
    data_png = (data - hist_min) / (hist_max - hist_min) * 255
    data_png[data_png < 0] = 0
    data_png[data_png > 255] = 255
    return data_png

def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]
    return right - left - 1

def np_ExtractBreast(img):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)
    h, w = img.shape

    y_center = h // 2
    y_range = int(h * 0.4)
    vertical_roi = img[y_center-y_range:y_center+y_range, :]
    b_arr = vertical_roi.std(axis=0) > 0  
    continuing = np_CountUpContinuingOnes(b_arr)
    x_indices = np.where(continuing == continuing.max())[0]
    x_min, x_max = x_indices[0], x_indices[-1]

    cropped_w = x_max - x_min + 1
    x_center = cropped_w // 2
    x_range = int(cropped_w * 0.4)
    horizontal_roi = img[:, x_min+x_center-x_range:x_min+x_center+x_range]
    b_arr = horizontal_roi.std(axis=1) > 0
    continuing = np_CountUpContinuingOnes(b_arr)
    y_indices = np.where(continuing == continuing.max())[0]
    y_min, y_max = y_indices[0], y_indices[-1]

    return img_copy[y_min:y_max+1, x_min:x_max+1], x_min, y_min

def resize_to_shortest_edge(img, target_size=224):
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = target_size, int(w * (target_size / h))
    else:
        new_h, new_w = int(h * (target_size / w)), target_size
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img, (w, h), (new_w, new_h)


def resize_to_fixed_size(img, target_size=(224, 224)):
    orig_h, orig_w = img.shape[:2]
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img, (orig_w, orig_h), target_size

def yolo_to_center_coordinates(x_center, y_center, image_width, image_height):
    x_center = int(x_center * image_width)
    y_center = int(y_center * image_height)
    return x_center, y_center

def crop_centered_box(img, x_center, y_center, width, height, view):
    img_width, img_height = img.size
    crop_left = x_center - width // 2
    crop_right = x_center + width // 2
    crop_top = y_center - height // 2
    crop_bottom = y_center + height // 2
    if crop_left < 0:
        crop_left = 0
        crop_right = width
    if crop_right > img_width:
        crop_right = img_width
        crop_left = img_width - width
    if crop_top < 0:
        crop_top = 0
        crop_bottom = height
    if crop_bottom > img_height:
        crop_bottom = img_height
        crop_top = img_height - height

    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(img_width, crop_left + width)
    crop_bottom = min(img_height, crop_top + height)

    if crop_right - crop_left == width and crop_bottom - crop_top == height:
        return img.crop((crop_left, crop_top, crop_right, crop_bottom))

    result = Image.new('RGB', (width, height), (0, 0, 0))
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    paste_y = max(0, (height // 2) - (y_center - crop_top))
    if view.startswith('L'):
        paste_x = 0
    else:
        paste_x = width - (crop_right - crop_left)
    result.paste(cropped, (paste_x, paste_y))
    return result


