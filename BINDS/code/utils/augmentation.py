from torchvision import transforms
from torchvision.transforms.functional import resize
import random
import torch
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
import numpy as np
import cv2

def mri_aug(imgs):
    rotate_time = random.randint(0, 3)
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = torch.rot90(imgs[i], rotate_time, [2, 3])
        imgs[i] = transforms.functional.adjust_brightness(imgs[i], bright_intensity)
        imgs[i] = transforms.functional.adjust_contrast(imgs[i], contrast_intensity)
    return imgs

def mammogram_aug(imgs):
    augmented_imgs = []
    for img in imgs:
        brightness_factor = random.uniform(0.5, 1.5)
        contrast_factor = random.uniform(0.5, 1.5)
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            angle = random.choice([0, 90, 180, 270])
            img = transforms.functional.rotate(img, angle)
        width, height = img.size
        if random.random() < 0.5:
            crop_ratio = random.uniform(0.8, 1.0)
            crop_height = int(height * crop_ratio)
            crop_width = int(width * crop_ratio)
            img = transforms.functional.center_crop(img, (crop_height, crop_width))
        augmented_imgs.append(img)
    return augmented_imgs

def ultrasound_aug(imgs):
    augmented_imgs = []
    for img in imgs:
        brightness_factor = random.uniform(0.5, 1.5)
        contrast_factor = random.uniform(0.5, 1.5)
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            angle = random.choice([0, 90, 180, 270])
            img = transforms.functional.rotate(img, angle)
        width, height = img.size
        if random.random() < 0.5:
            crop_ratio = random.uniform(0.8, 1.0)
            crop_height = int(height * crop_ratio)
            crop_width = int(width * crop_ratio)
            img = transforms.functional.center_crop(img, (crop_height, crop_width))
        augmented_imgs.append(img)
    return augmented_imgs

def pathology_aug(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(8, 12) / 10.0
        contrast_intensity = random.randint(8, 12) / 10.0
        rotate_time = random.randint(0, 3)
        imgs[i] = torch.rot90(imgs[i], rotate_time, [1, 2])
        imgs[i] = transforms.functional.adjust_brightness(imgs[i], bright_intensity)
        imgs[i] = transforms.functional.adjust_contrast(imgs[i], contrast_intensity)
    return imgs

def pathology_aug_hed(imgs, shift_range_h=(0.3, 1.8), shift_range_e=(0.3, 1.8), shift_range_d=(0.5, 1.5)):
    for i in range(len(imgs)):
        image_np = imgs[i].permute(1, 2, 0).cpu().numpy()
        hed_image = rgb2hed(image_np)
        random_shift_h = np.random.uniform(shift_range_h[0], shift_range_h[1])
        random_shift_e = np.random.uniform(shift_range_e[0], shift_range_e[1])
        random_shift_d = np.random.uniform(shift_range_d[0], shift_range_d[1])
        hed_image[:, :, 0] *= random_shift_h
        hed_image[:, :, 1] *= random_shift_e
        hed_image[:, :, 2] *= random_shift_d
        rgb_image = hed2rgb(hed_image)
        rgb_image = np.clip(rgb_image, 0, 1)
        imgs[i] = torch.tensor(rgb_image).permute(2, 0, 1).float()  # Convert back to tensor and (C, H, W)
    return imgs

def pathology_aug_hsv(imgs, hue_shift_limit=(-0.1, 0.1), sat_shift_limit=(-0.1, 0.1), val_shift_limit=(-0.1, 0.1)):
    for i in range(len(imgs)):
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1]) * 180
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1]) * 255
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1]) * 255
        image_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + sat_shift, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + val_shift, 0, 255)
        rgb_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
        imgs[i] = torch.tensor(rgb_image).float().permute(2, 0, 1) / 255.0  # Convert back to tensor and (C, H, W)
    return imgs