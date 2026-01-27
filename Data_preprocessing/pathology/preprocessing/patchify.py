from xml.etree.ElementTree import parse
from glob import glob
from torchvision.utils import draw_segmentation_masks as draw_mask
import torchvision.transforms.functional as F
import torch
import tqdm
import os
from einops import repeat,rearrange
from filter import *
import openslide
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
from utils.plot import *
import json

class dataset_info:
    def __init__(self):
        with open('../dataset/slide.json', 'r') as file:
            self.svs_path = json.load(file)
        self.base_dir = 'F:/patch/'
        self.wsi_base_dir = 'F:/raw/'
        os.makedirs(self.base_dir, exist_ok=True)
        self.slide_dir = None
        self.svs = None
        self.slide = None
        self.mask = None
        self.svs_name = None

        self.w_1x = None # the resolution of the WSI in 1x magnification
        self.h_1x = None
        self.w_40x = None  # the resolution of the WSI in 40x magnification
        self.h_40x = None

        self.threshold = 50 # tissue_rate threshold (lower than the threshold: discard the patch)
        self.patch_w_1x = 64 # patch size in 1x magnification
        self.patch_h_1x = 64

        self.patch_w_40x = self.patch_w_1x * 16 # patch size in 40x magnification
        self.patch_h_40x = self.patch_w_1x * 16
        self.img_size = (224, 224)  # resize the cropped patch in 40x magnification
        self.max_patch_num = 4000 # the maximum number of patches in each WSI

    def makedir(self):
        self.slide_dir = self.base_dir + self.id + '/' + self.name + '/'
        self.patch_dir = self.base_dir + self.id + '/' + self.name + '/medium/'
        print('slide dir:', self.slide_dir, self.patch_dir)
        os.makedirs(self.slide_dir, exist_ok=True)
        os.makedirs(self.patch_dir, exist_ok=True)

def read_wsi(info):
    print("load svs...........")
    info.path = glob(info.wsi_base_dir + '/**/' + info.name + '.svs', recursive=True)[0]
    print('svs path:', info.path)
    info.slide = openslide.OpenSlide(info.path)
    info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    info.num_row_patch = math.floor(info.h_1x / info.patch_h_1x)
    info.num_col_patch = math.floor(info.w_1x / info.patch_w_1x)
    thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    thumbnail.save(info.slide_dir + 'thumbnail' + '_' + str(info.num_row_patch) + '_' + str(info.num_col_patch) + '.png')
    slide = np.asarray(thumbnail)
    info.slide_np = np.clip(slide, 0, 255).astype(np.uint8)

def get_patch(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.patch_w_40x, info.patch_h_40x))
    patch = patch.resize(info.img_size)
    return patch

def otsu_threshold(info):
    print('perform otsu threshold..................')
    grayscale = filter_rgb_to_grayscale(info.slide_np)
    complement = filter_complement(grayscale)
    filtered = filter_otsu_threshold(complement)
    mask = Image.fromarray(filtered)
    mask.save(info.slide_dir + 'mask.png')
    info.mask = np.clip(filtered, 0, 1)
    return

def get_tissue_rate(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    mask = info.mask[x:x+info.patch_w_1x,y:y+info.patch_h_1x]
    tissue_rate = int(np.sum(mask) * 100.0 / (mask.shape[0] * mask.shape[1]))
    return tissue_rate

def grid_wsi(info):
    print("grid and select patch from whole slide image................")
    print('num row:', info.num_row_patch, 'num col:', info.num_col_patch)
    count = 0
    for col_id in range(4, info.num_col_patch - 4, 1):
        for row_id in range(4, info.num_row_patch - 4, 1):
            tissue_rate = get_tissue_rate(row_id, col_id, info)
            if tissue_rate > info.threshold:
                patch = get_patch(row_id, col_id, info)
                patch.save(info.patch_dir + str(row_id) + '_' + str(col_id) + '.png')
                count += 1
                if count > info.max_patch_num:
                    return

if __name__ == '__main__':
    info = dataset_info()
    for svs_path in tqdm(info.svs_path):
        info.id = svs_path[0]
        info.name = svs_path[1]
        info.makedir()
        print('preprocessing whole slide image:', svs_path)
        read_wsi(info)
        otsu_threshold(info)
        grid_wsi(info)