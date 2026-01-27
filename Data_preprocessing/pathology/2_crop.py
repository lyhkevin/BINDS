from xml.etree.ElementTree import parse
from glob import glob
from torchvision.utils import draw_segmentation_masks as draw_mask
import torchvision.transforms.functional as F
import torch
import tqdm
import os
from einops import repeat, rearrange
import json
from torchvision import transforms
import openslide
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import math
from torch.utils.data import Dataset, DataLoader

class dataset_info:
    def __init__(self):
        self.mask_dir = './seg/'
        self.save_dir = './patches/'
        self.wsi_base_dir = '../example_data_and_weight/pathology/data'
        os.makedirs(self.save_dir, exist_ok=True)

        self.slide_dir = None
        self.svs = None
        self.slide = None
        self.mask = None
        self.svs_name = None
        self.path = None
        self.name = None

        self.w_1x = None
        self.h_1x = None
        self.w_40x = None
        self.h_40x = None

        self.num_row_patch = None
        self.num_col_patch = None
        self.patch_dir = None

        self.patch_w_1x = None
        self.patch_h_1x = None
        self.patch_w_40x = None
        self.patch_h_40x = None

        self.tumor_threshold_small = 80
        self.tissue_threshold_small = 80
        
        self.tumor_threshold_medium = 60
        self.tissue_threshold_medium = 60
        
        self.tumor_threshold_large = 40
        self.tissue_threshold_large = 40
        
        self.tumor_threshold = None
        self.tissue_threshold = None

        self.img_size = (512, 512)
        self.max_patch_num = 250

        self.scales_config = [
            {'name': 'small',  'w_1x': 64,  'h_1x': 64},
            {'name': 'medium', 'w_1x': 128, 'h_1x': 128},
            {'name': 'large',  'w_1x': 256, 'h_1x': 256}
        ]

        self.svs_path = []
        mask_paths = glob(self.mask_dir + '/*')
        for mp in mask_paths:
            name = os.path.basename(mp)
            found_svs = glob(self.wsi_base_dir + '/**/' + name + '.svs', recursive=True)
            if found_svs:
                self.svs_path.append(found_svs[0])

    def load_wsi_and_mask(self):
        print(f"Loading SVS: {self.path}")
        self.slide = openslide.OpenSlide(self.path)
        self.w_1x, self.h_1x = self.slide.level_dimensions[2]

        mask_path = os.path.join(self.mask_dir, self.name, 'segmentation_mask.npy')
        otsu_path = os.path.join(self.mask_dir, self.name, 'otsu_mask.npy')

        if os.path.exists(mask_path):
            self.seg_mask = np.load(mask_path)
        else:
            self.seg_mask = None

        if os.path.exists(otsu_path):
            self.mask = np.load(otsu_path)
        else:
            self.mask = None

    def update_scale_info(self, scale_cfg):
        scale_name = scale_cfg['name']
        self.patch_w_1x = scale_cfg['w_1x']
        self.patch_h_1x = scale_cfg['h_1x']

        self.patch_w_40x = self.patch_w_1x * 16
        self.patch_h_40x = self.patch_h_1x * 16

        if scale_name == 'small':
            self.tumor_threshold = self.tumor_threshold_small
            self.tissue_threshold = self.tissue_threshold_small
        elif scale_name == 'medium':
            self.tumor_threshold = self.tumor_threshold_medium
            self.tissue_threshold = self.tissue_threshold_medium
        elif scale_name == 'large':
            self.tumor_threshold = self.tumor_threshold_large
            self.tissue_threshold = self.tissue_threshold_large

        self.patch_dir = os.path.join(self.save_dir, self.name, scale_name)
        os.makedirs(self.patch_dir, exist_ok=True)

        self.num_row_patch = math.floor(self.h_1x / self.patch_h_1x)
        self.num_col_patch = math.floor(self.w_1x / self.patch_w_1x)

def get_patch(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.patch_w_40x, info.patch_h_40x))
    patch = patch.convert('RGB')
    patch = patch.resize(info.img_size)
    return patch

def grid_wsi(info):
    count = 0
    for row_id in range(info.num_row_patch):
        for col_id in range(info.num_col_patch):
            x = row_id * info.patch_w_1x
            y = col_id * info.patch_h_1x

            seg_mask_patch = info.seg_mask[x : x + info.patch_w_1x, y : y + info.patch_h_1x]
            otsu_mask_patch = info.mask[x : x + info.patch_w_1x, y : y + info.patch_h_1x]

            if seg_mask_patch.size == 0 or otsu_mask_patch.size == 0:
                continue

            tumor_rate = int(np.sum(seg_mask_patch) * 100.0 / seg_mask_patch.size)
            foreground_rate = int(np.sum(otsu_mask_patch) * 100.0 / otsu_mask_patch.size)

            if tumor_rate >= info.tumor_threshold and foreground_rate >= info.tissue_threshold:
                patch = get_patch(row_id, col_id, info)
                save_path = os.path.join(info.patch_dir, f"{row_id}_{col_id}.png")
                patch.save(save_path)

                count += 1
                if count >= info.max_patch_num:
                    return

if __name__ == '__main__':
    info = dataset_info()

    if not info.svs_path:
        exit()

    for svs_path in tqdm(info.svs_path):
        info.path = svs_path
        info.name = os.path.splitext(os.path.basename(svs_path))[0]

        try:
            info.load_wsi_and_mask()

            if info.seg_mask is None or info.mask is None:
                continue

            for scale_cfg in info.scales_config:
                info.update_scale_info(scale_cfg)
                grid_wsi(info)

        except Exception as e:
            print(e)
        finally:
            if info.slide:
                info.slide.close()