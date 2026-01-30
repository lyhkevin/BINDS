import os
import numpy as np
from PIL import Image
from torchvision import transforms
import random
import torch.utils.data as data
from glob import glob
import pandas as pd
import nibabel as nib
import json
from utils.augmentation import *
from utils.util import *

def repeat_elements(input_list, n=10):
    repeated_list = input_list.copy()
    while len(repeated_list) < n:
        element = random.choice(input_list)
        repeated_list.append(element)
    return repeated_list

class Dataset(data.Dataset):
    def __init__(self, mode, opt):
        self.opt = opt
        self.epoch = 0
        self.epoch_alignment = opt.epoch_alignment
        self.alignment = opt.alignment
        self.multimodal = opt.multimodal

        self.modalities = opt.modalities
        self.key_modalities = opt.key_modalities
        self.radiological = opt.radiological
        self.mri_view = opt.mri_view
        self.mri_modalities = opt.mri_modalities
        self.num_mri_modalities = len(self.mri_modalities)
        self.pathology_size = opt.pathology_size
        self.mri_size = opt.mri_size
        self.mammogram_size = opt.mammogram_size
        self.ultrasound_size = opt.ultrasound_size
        self.pathology_scale = opt.pathology_scale

        self.benign_ultrasound_root = opt.wenzhou_root
        self.yunnan_benign_clinical = opt.yunnan_benign_clinical
        self.yunnan_benign_clinical = pd.read_excel(self.yunnan_benign_clinical)
        self.yunnan_benign_root = opt.yunnan_benign_root

        self.yunnan_neoadjuvant_clinical = opt.yunnan_neoadjuvant_clinical
        self.yunnan_neoadjuvant_clinical = pd.read_excel(self.yunnan_neoadjuvant_clinical)
        self.yunnan_neoadjuvant_root = opt.yunnan_neoadjuvant_root

        self.yunnan_surgical_clinical = opt.yunnan_surgical_clinical
        self.yunnan_surgical_clinical = pd.read_excel(self.yunnan_surgical_clinical)
        self.yunnan_surgical_root = opt.yunnan_surgical_root

        self.wenzhou_clinical = opt.wenzhou_clinical
        self.wenzhou_clinical = pd.read_excel(self.wenzhou_clinical)
        self.wenzhou_benign_root = opt.wenzhou_benign_root
        self.wenzhou_root = opt.wenzhou_root

        self.xinan_surgical_clinical = opt.xinan_surgical_clinical
        self.xinan_surgical_clinical = pd.read_excel(self.xinan_surgical_clinical)
        self.xinan_surgical_root = opt.xinan_surgical_root

        self.xinan_neoadjuvant_clinical = opt.xinan_neoadjuvant_clinical
        self.xinan_neoadjuvant_clinical = pd.read_excel(self.xinan_neoadjuvant_clinical)
        self.xinan_neoadjuvant_root = opt.xinan_neoadjuvant_root

        self.xiangya_clinical = opt.xiangya_clinical
        self.xiangya_clinical = pd.read_excel(self.xiangya_clinical)
        self.xiangya_root = opt.xiangya_root

        self.guizhou_clinical = opt.guizhou_clinical
        self.guizhou_clinical = pd.read_excel(self.guizhou_clinical)
        self.guizhou_root = opt.guizhou_root

        self.shanghai_clinical = opt.shanghai_clinical
        self.shanghai_clinical = pd.read_excel(self.shanghai_clinical)
        self.shanghai_root = opt.shanghai_root

        self.hangzhou_clinical = opt.hangzhou_clinical
        self.hangzhou_clinical = pd.read_excel(self.hangzhou_clinical)
        self.hangzhou_root = opt.hangzhou_root

        self.fuyiyuan_clinical = opt.fuyiyuan_clinical
        self.fuyiyuan_clinical = pd.read_excel(self.fuyiyuan_clinical)
        self.fuyiyuan_root = opt.fuyiyuan_root

        self.public_root = opt.public_root

        self.augmentation = opt.augmentation
        self.augmentation_pathology = opt.augmentation_pathology
        self.target = opt.target
        self.clinical = opt.clinical
        self.oversample = opt.oversample
        self.oversample_targets = opt.oversample_targets
        self.oversample_rates = opt.oversample_rates
        self.oversample_modality = opt.oversample_modality
        self.num_classes = get_num_class(self.target)
        self.samples = []
        self.valid_values = get_valid_values(self.target)
        self.random_mask = opt.random_mask
        self.mode = mode

        print('----------------------------------------------------')
        print('mode:', self.mode)

        print('load Yunnan........')
        self.samples_yunnan = []
        self.load_yunnan()
        self.stats(self.samples_yunnan)
        print()

        if opt.split_data == True or (self.mode == 'train' or self.multimodal == False):
            print('load Wenzhou........')
            self.samples_wenzhou = []
            self.load_wenzhou()
            self.stats(self.samples_wenzhou)
            print()

        if opt.split_data == True or (self.mode == 'train' and self.multimodal == False):
            print('load Wenzhou benign........')
            self.samples_wenzhou_benign = []
            self.load_wenzhou_benign()
            self.stats(self.samples_wenzhou_benign)
            print()

        if opt.split_data == True or (self.mode == 'train' and self.multimodal == False):
            print('load Public........')
            self.samples_public = []
            self.load_public()
            self.stats(self.samples_public)
            print()

        if opt.split_data == True or 'external' in self.mode:
            print('load Xiangya........')
            self.samples_xiangya = []
            self.load_external(self.xiangya_clinical, self.xiangya_root, 'xiangya')
            self.stats(self.samples_xiangya)
            print()

            print('load Guizhou........')
            self.samples_guizhou = []
            self.load_external(self.guizhou_clinical, self.guizhou_root, 'guizhou')
            self.stats(self.samples_guizhou)
            print()

            print('load Shanghai........')
            self.samples_shanghai = []
            self.load_external(self.shanghai_clinical, self.shanghai_root, 'shanghai')
            self.stats(self.samples_shanghai)
            print()

            print('load Fuyiyuan........')
            self.samples_fuyiyuan = []
            self.load_external(self.fuyiyuan_clinical, self.fuyiyuan_root, 'fuyiyuan')
            self.stats(self.samples_fuyiyuan)
            print()

            print('load Hangzhou........')
            self.samples_hangzhou = []
            self.load_external(self.hangzhou_clinical, self.hangzhou_root, 'hangzhou')
            self.stats(self.samples_hangzhou)
            print()

            print('load Xinan surgical........')
            self.samples_xinan_surgical = []
            self.load_external(self.xinan_surgical_clinical, self.xinan_surgical_root, 'xinan_surgical')
            self.stats(self.samples_xinan_surgical)
            print()

            print('load Xinan neoadjuvant........')
            self.samples_xinan_neoadjuvant = []
            self.load_external(self.xinan_neoadjuvant_clinical, self.xinan_neoadjuvant_root, 'xinan_neoadjuvant')
            self.stats(self.samples_xinan_neoadjuvant)
            print()

        print('Stats of the entire dataset')
        self.stats(self.samples)

        if opt.split_data == True:
            return
        with open(opt.data_split_path, 'r') as file:
            self.data_split = json.load(file)
        print('split data')
        self.split_data()

        if self.mode == 'train':
            print('Train samples before oversampling:')
            self.stats(self.train_samples)
            if self.alignment == True:
                self.samples_alignment = []
                for sample in self.train_samples:
                    if 'yunnan_neoadjuvant' in sample['cohort']:
                        if sample['pathology'] is not None:
                            flag = True
                            for key_modality in self.key_modalities:
                                if sample[key_modality] is None:
                                    flag = False
                                    break
                            if flag == True:
                                self.samples_alignment.append(sample)
                print('Train samples before oversampling for alignment:')
                self.stats(self.samples_alignment)
            if self.oversample == True:
                self.oversample_training_set()
                print('Train samples after oversampling:')
                self.stats(self.train_samples)
                if self.alignment == True:
                    print('Train samples after oversampling for alignment:')
                    self.stats(self.samples_alignment)
        if self.mode == 'validation':
            print('Validation samples before oversampling:')
            self.stats(self.validation_samples)
        if self.mode == 'internal':
            print('Internal samples before oversampling:')
            self.stats(self.internal_samples)
        if self.mode == 'external_1':
            print('External_1 samples before oversampling:')
            self.stats(self.external_1_samples)
        if self.mode == 'external_2':
            print('External_2 samples before oversampling:')
            self.stats(self.external_2_samples)
        if self.mode == 'reader':
            print('Reader study samples before oversampling:')
            self.stats(self.reader_samples)

        self.ultrasound_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.ultrasound_size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.mammogram_transform = transforms.Compose(
            [transforms.Resize(self.mammogram_size, Image.NEAREST), transforms.ToTensor()])
        self.pathology_transform = transforms.Compose(
            [transforms.Resize(self.pathology_size, Image.NEAREST), transforms.ToTensor()])

    def examine_sample(self, sample):
        if self.radiological == True and sample['mri'] is None and sample['mammogram'] is None and sample['ultrasound'] is None:
            return False
        num_modality = 0
        for modality in self.modalities:
            if sample[modality] is not None:
                num_modality += 1
        if num_modality < self.opt.min_num_modalities:
            return False
        for modality in self.key_modalities:
            if sample[modality] is None:
                return False
        if all(element == -1 for element in sample['labels']):
            return False
        return True

    def load_wenzhou_benign(self):
        subjects = glob(self.wenzhou_benign_root + '*')
        count = 1
        for subject in subjects:
            labels = []
            clinicals = []
            for target in self.target:
                if target == 'Risk':
                    labels.append(0)
                else:
                    labels.append(-1)
            for clinical in self.clinical:
                clinicals.append(-1)
            sample = {'cohort': 'wenzhou_benign', 'id': count, 'ultrasound': subject, 'mammogram': None, 'mri': None,
                      'pathology': None, 'labels': labels, 'clinical': clinicals}
            if self.examine_sample(sample):
                self.samples.append(sample)
                self.samples_wenzhou_benign.append(sample)
                count = count + 1

    def load_wenzhou(self):
        for i, row in self.wenzhou_clinical.iterrows():
            id = int(row['ID'])
            labels = []
            clinicals = []
            for target in self.target:
                if target not in self.wenzhou_clinical.columns:
                    labels.append(-1)
                    continue
                if row[target] not in self.valid_values[target]:
                    labels.append(-1)
                else:
                    labels.append(int(row[target]))
            for clinical in self.clinical:
                if clinical not in self.wenzhou_clinical.columns or pd.isna(row[clinical]):
                    clinicals.append(-1)
                else:
                    clinicals.append(row[clinical])
            if os.path.exists(self.wenzhou_root + str(id)):
                sample = {'cohort': 'wenzhou', 'id': id, 'ultrasound': self.wenzhou_root + str(id), 'mammogram': None,
                          'mri': None,
                          'pathology': None, 'labels': labels, 'clinical': clinicals}
                if self.examine_sample(sample):
                    self.samples.append(sample)
                    self.samples_wenzhou.append(sample)

    def load_public(self):
        subjects = glob(self.public_root + '*/*/*')
        count = 1
        for subject in subjects:
            labels = []
            clinicals = []
            for target in self.target:
                if target == 'Risk':
                    labels.append(int(subject[-1]))
                else:
                    labels.append(-1)
            for clinical in self.clinical:
                clinicals.append(-1)
            sample = {'cohort': 'public', 'id': count, 'ultrasound': None, 'mammogram': None, 'mri': None,
                      'pathology': None, 'labels': labels, 'clinical': clinicals}
            if 'mammogram' in subject:
                sample['mammogram'] = subject
            else:
                sample['ultrasound'] = subject
            if self.examine_sample(sample):
                self.samples.append(sample)
                self.samples_public.append(sample)
                count = count + 1

    def load_yunnan(self):
        self.load_yunnan_benign()
        self.load_yunnan_neoadjuvant()
        self.load_yunnan_surgical()

    def load_yunnan_benign(self):
        for i, row in self.yunnan_benign_clinical.iterrows():
            id = int(row['ID'])
            labels = []
            clinicals = []
            for target in self.target:
                if target == 'Risk':
                    labels.append(0)
                else:
                    labels.append(-1)
            for clinical in self.clinical:
                if clinical not in self.yunnan_benign_clinical.columns or pd.isna(row[clinical]):
                    clinicals.append(-1)
                else:
                    clinicals.append(row[clinical])
            sample = {'cohort': 'yunnan_benign', 'id': id, 'ultrasound': None, 'mammogram': None, 'mri': None,
                      'pathology': None, 'labels': labels, 'clinical': clinicals}
            for modality in ['ultrasound', 'mammogram']:
                modality_path = self.yunnan_benign_root + modality + '/' + str(id)
                if os.path.exists(modality_path):
                    sample[modality] = modality_path
            if os.path.exists(self.yunnan_benign_root + 'mri/' + str(id)):
                mri_modality = True
                for modality in self.mri_modalities:
                    if not os.path.exists(self.yunnan_benign_root + 'mri/' + str(id) + '/' + modality + '.nii.gz'):
                        mri_modality = False
                if mri_modality == True:
                    sample['mri'] = self.yunnan_benign_root + 'mri/' + str(id)
            if self.examine_sample(sample):
                self.samples.append(sample)
                self.samples_yunnan.append(sample)

    def load_yunnan_neoadjuvant(self):
        count = 0
        for i, row in self.yunnan_neoadjuvant_clinical.iterrows():
            id = int(row['ID'])
            labels = []
            clinicals = []
            for target in self.target:
                if target == 'Risk':
                    labels.append(1)
                else:
                    if target not in self.yunnan_neoadjuvant_clinical.columns:
                        labels.append(-1)
                        continue
                    if row[target] not in self.valid_values[target]:
                        labels.append(-1)
                    else:
                        labels.append(int(row[target]))
            for clinical in self.clinical:
                if clinical not in self.yunnan_neoadjuvant_clinical.columns or pd.isna(row[clinical]):
                    clinicals.append(-1)
                else:
                    clinicals.append(row[clinical])
            sample = {'cohort': 'yunnan_neoadjuvant', 'id': id, 'ultrasound': None, 'mammogram': None, 'mri': None,
                      'pathology': None, 'labels': labels, 'clinical': clinicals}
            for modality in ['ultrasound', 'mammogram']:
                modality_path = self.yunnan_neoadjuvant_root + modality + '/' + str(id)
                if os.path.exists(modality_path):
                    sample[modality] = modality_path
            if os.path.exists(self.yunnan_neoadjuvant_root + 'mri/' + str(id)):
                mri_modality = True
                for modality in self.mri_modalities:
                    if not os.path.exists(self.yunnan_neoadjuvant_root + 'mri/' + str(id) + '/' + modality + '.nii.gz'):
                        mri_modality = False
                if mri_modality == True:
                    sample['mri'] = self.yunnan_neoadjuvant_root + 'mri/' + str(id)
            pathology_paths = glob(self.yunnan_neoadjuvant_root + 'pathology/' + str(id) + '/*')
            path_pathology = []
            if len(pathology_paths) > 0 and 'pathology' in self.modalities:
                for pathology_path in pathology_paths:
                    scales = glob(pathology_path + '/*')
                    if len(scales) == 3:
                        path_pathology.append(pathology_path)
                if len(path_pathology) > 0:
                    sample['pathology'] = path_pathology
                    count += 1
            if self.examine_sample(sample):
                self.samples.append(sample)
                self.samples_yunnan.append(sample)

    def load_yunnan_surgical(self):
        for i, row in self.yunnan_surgical_clinical.iterrows():
            id = int(row['ID'])
            labels = []
            clinicals = []
            for target in self.target:
                if target == 'Risk':
                    labels.append(1)
                else:
                    if target not in self.yunnan_surgical_clinical.columns:
                        labels.append(-1)
                        continue
                    if row[target] not in self.valid_values[target]:
                        labels.append(-1)
                    else:
                        labels.append(int(row[target]))
            for clinical in self.clinical:
                if clinical not in self.yunnan_surgical_clinical.columns or pd.isna(row[clinical]):
                    clinicals.append(-1)
                else:
                    clinicals.append(row[clinical])
            sample = {'cohort': 'yunnan_surgical', 'id': id, 'ultrasound': None, 'mammogram': None, 'mri': None,
                      'pathology': None, 'labels': labels, 'clinical': clinicals}
            for modality in ['ultrasound', 'mammogram']:
                modality_path = self.yunnan_surgical_root + modality + '/' + str(id)
                if os.path.exists(modality_path):
                    sample[modality] = modality_path
            if os.path.exists(self.yunnan_surgical_root + 'mri/' + str(id)):
                mri_modality = True
                for modality in self.mri_modalities:
                    if not os.path.exists(self.yunnan_surgical_root + 'mri/' + str(id) + '/' + modality + '.nii.gz'):
                        mri_modality = False
                if mri_modality == True:
                    sample['mri'] = self.yunnan_surgical_root + 'mri/' + str(id)
            if self.examine_sample(sample):
                self.samples.append(sample)
                self.samples_yunnan.append(sample)

    def load_external(self, clinical_external, root, cohort):
        for i, row in clinical_external.iterrows():
            id = str(int(row['ID']))
            if 'xinan' in cohort:
                id = id.zfill(10)
            labels = []
            clinicals = []
            for target in self.target:
                if target not in clinical_external.columns:
                    if target == 'Risk':
                        labels.append(1)
                    else:
                        labels.append(-1)
                    continue
                if row[target] not in self.valid_values[target]:
                    labels.append(-1)
                else:
                    labels.append(int(row[target]))
            for clinical in self.clinical:
                if clinical not in clinical_external.columns or pd.isna(row[clinical]):
                    clinicals.append(-1)
                else:
                    clinicals.append(row[clinical])
            sample = {'cohort': cohort, 'id': id, 'ultrasound': None, 'mammogram': None, 'mri': None, 'pathology': None,
                      'labels': labels, 'clinical': clinicals}
            for modality in ['ultrasound', 'mammogram']:
                modality_path = root + modality + '/' + str(id)
                if os.path.exists(modality_path):
                    sample[modality] = modality_path
            if os.path.exists(root + 'mri/' + str(id)):
                mri_modality = True
                for modality in self.mri_modalities:
                    if not os.path.exists(root + 'mri/' + str(id) + '/' + modality + '.nii.gz'):
                        mri_modality = False
                if mri_modality == True:
                    sample['mri'] = root + 'mri/' + str(id)
            if self.examine_sample(sample):
                self.samples.append(sample)
                if cohort == 'xiangya':
                    self.samples_xiangya.append(sample)
                if cohort == 'guizhou':
                    self.samples_guizhou.append(sample)
                if cohort == 'shanghai':
                    self.samples_shanghai.append(sample)
                if cohort == 'fuyiyuan':
                    self.samples_fuyiyuan.append(sample)
                if cohort == 'hangzhou':
                    self.samples_hangzhou.append(sample)
                if cohort == 'xinan_surgical':
                    self.samples_xinan_surgical.append(sample)
                if cohort == 'xinan_neoadjuvant':
                    self.samples_xinan_neoadjuvant.append(sample)

    def stats(self, samples):
        num_modalities = {'ultrasound': 0, 'mammogram': 0, 'mri': 0, 'pathology': 0}
        stats = get_count(self.target)
        combo_counts = {}
        modalities_order = ['ultrasound', 'mammogram', 'mri', 'pathology']

        for sample in samples:
            present = []
            for modality in modalities_order:
                if sample.get(modality) is not None:
                    num_modalities[modality] += 1
                    present.append(modality)
            if present:
                combo = tuple(present)
                combo_counts[combo] = combo_counts.get(combo, 0) + 1
            for i in range(len(self.target)):
                stats[i][sample['labels'][i]] += 1
        for i in range(len(self.target)):
            print(self.target[i], stats[i])
        print('Num modalities:', num_modalities)
        if combo_counts:
            print('Modality combinations (count > 0):')
            for combo in sorted(combo_counts.keys(), key=lambda x: (len(x), x)):
                print(f"{combo}: {combo_counts[combo]}")

    def split_data(self):
        self.train_samples = []
        self.validation_samples = []
        self.internal_samples = []
        self.external_1_samples = []
        self.external_2_samples = []
        self.reader_samples = []
        for sample in self.samples:
            if [sample['cohort'], sample['id']] in self.data_split["train"]:
                self.train_samples.append(sample)
            if [sample['cohort'], sample['id']] in self.data_split["validation"]:
                self.validation_samples.append(sample)
            if [sample['cohort'], sample['id']] in self.data_split["internal"]:
                self.internal_samples.append(sample)
            if [sample['cohort'], sample['id']] in self.data_split["external_1"]:
                self.external_1_samples.append(sample)
            if [sample['cohort'], sample['id']] in self.data_split["external_2"]:
                self.external_2_samples.append(sample)
            if 'reader' in self.data_split:
                if [sample['cohort'], sample['id']] in self.data_split["reader"]:
                    self.reader_samples.append(sample)

    def oversample_training_set(self):
        def modality_key(sample):
            present = []
            if sample.get('ultrasound') is not None:
                present.append('us')
            if sample.get('mammogram') is not None:
                present.append('mm')
            if sample.get('mri') is not None:
                present.append('mri')
            if not present:
                return 'none'
            order = {'us': 0, 'mm': 1, 'mri': 2}
            present.sort(key=lambda x: order[x])
            return '+'.join(present)

        if self.oversample == True:
            train_samples = []
            for sample in self.train_samples:
                train_samples.append(sample)
                for i, label in enumerate(sample['labels']):
                    if label != -1:
                        for j in range(self.oversample_rates[i][label]):
                            train_samples.append(sample)
                if self.multimodal == True:
                    key = modality_key(sample)
                    repeat = int(self.oversample_modality.get(key, 0))
                    for _ in range(max(0, repeat)):
                        train_samples.append(sample)
            self.train_samples = train_samples

    def load_ultrasound(self, path):
        imgs_ultrasound = []
        ultrasound_path = glob(path + '/*.png')
        if len(ultrasound_path) == 1:
            ultrasound_path = ultrasound_path * 2
        for path in ultrasound_path:
            img = Image.open(path).convert('RGB')
            imgs_ultrasound.append(img)
        if self.augmentation and self.mode == 'train':
            imgs_ultrasound = ultrasound_aug(imgs_ultrasound)
        imgs_ultrasound = [self.ultrasound_transform(img) for img in imgs_ultrasound]
        return imgs_ultrasound

    def load_mammogram(self, path):
        imgs_mammogram = []
        mammogram_path = glob(path + '/tumor/*.png')
        if len(mammogram_path) == 1:
            mammogram_path = mammogram_path * 2
        for path in mammogram_path:
            img = Image.open(path).convert('RGB')
            imgs_mammogram.append(img)
        if self.augmentation == True and self.mode == 'train':
            imgs_mammogram = mammogram_aug(imgs_mammogram)
        imgs_mammogram = [self.mammogram_transform(img) for img in imgs_mammogram]
        return imgs_mammogram

    def load_mri(self, path):
        imgs_mri = []
        for modality in self.mri_modalities:
            img = nib.load(path + '/' + modality + '.nii.gz')
            img = img.get_fdata().astype(np.float32)
            if self.mri_view == 'Axial':
                img = np.transpose(img, (2, 1, 0))
            if self.mri_view == 'Coronal':
                img = np.transpose(img, (1, 2, 0))
            img = np.flip(img, axis=1).copy()
            img = np.flip(img, axis=2).copy()
            img = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
            img = torch.nn.functional.interpolate(img, size=self.mri_size, mode='trilinear', align_corners=False).squeeze()
            img = img.unsqueeze(dim=1)
            img = img.repeat(1, 3, 1, 1)
            imgs_mri.append(img)
        if self.augmentation == True and self.mode == 'train':
            imgs_mri = mri_aug(imgs_mri)
        return imgs_mri

    def load_pathology(self, path):
        imgs_pathology = []
        small_path = glob(path + '/small/*.png')
        medium_path = glob(path + '/medium/*.png')
        large_path = glob(path + '/large/*.png')
        if len(small_path) < 10:
            small_path = repeat_elements(small_path)
        if len(medium_path) < 10:
            medium_path = repeat_elements(medium_path)
        if len(large_path) < 10:
            large_path = repeat_elements(large_path)
        pathology_path = small_path + medium_path + large_path
        for path in pathology_path:
            img = Image.open(path).convert('RGB')
            img = self.pathology_transform(img)
            imgs_pathology.append(img)
        if self.augmentation == True and self.mode == 'train':
            if self.augmentation_pathology == 'hed':
                imgs_pathology = pathology_aug_hed(imgs_pathology)
            elif self.augmentation_pathology == 'hsv':
                imgs_pathology = pathology_aug_hsv(imgs_pathology)
            else:
                imgs_pathology = pathology_aug(imgs_pathology)
        return imgs_pathology

    def random_masking(self, sample):
        modalities = ['ultrasound', 'mammogram', 'mri']
        available_modalities = [m for m in modalities if sample.get(m) is not None]
        n = len(available_modalities)
        if n < 2:
            return []
        if random.random() < 1 - self.opt.mask_ratio:
            return []
        if n == 2:
            mask_num = 1
        else:
            mask_num = random.choice([1, 2])
        return random.sample(available_modalities, mask_num)

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.alignment == True and self.epoch < self.epoch_alignment:
                sample = self.samples_alignment[index]
            else:
                sample = self.train_samples[index]
        if self.mode == 'validation':
            sample = self.validation_samples[index]
        if self.mode == 'internal':
            sample = self.internal_samples[index]
        if self.mode == 'external_1':
            sample = self.external_1_samples[index]
        if self.mode == 'external_2':
            sample = self.external_2_samples[index]
        if self.mode == 'reader':
            sample = self.reader_samples[index]

        load_sample = {'cohort': sample['cohort'], 'id': sample['id'], 'has_ultrasound': None, 'has_mammogram': None,
                       'has_mri': None, 'has_pathology': None,
                       'ultrasound': None, 'mammogram': None, 'mri': None, 'pathology': None, 'labels': None,
                       'clinical': sample['clinical']}

        if self.random_mask == True and self.mode == 'train':
            masked_modality = self.random_masking(sample)
        else:
            masked_modality = []

        if 'ultrasound' in self.modalities:
            if sample['ultrasound'] != None and 'ultrasound' not in masked_modality:
                imgs_ultrasound = self.load_ultrasound(sample['ultrasound'])
                imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0)
                load_sample['has_ultrasound'] = True
                load_sample['ultrasound'] = imgs_ultrasound
            else:
                load_sample['has_ultrasound'] = False
                if 'ultrasound' in self.modalities:
                    load_sample['ultrasound'] = torch.zeros(2, 3, self.ultrasound_size[0], self.ultrasound_size[1])

        if 'mammogram' in self.modalities:
            if sample['mammogram'] != None and 'mammogram' not in masked_modality:
                imgs_mammogram = self.load_mammogram(sample['mammogram'])
                imgs_mammogram = torch.stack(imgs_mammogram, dim=0)
                load_sample['has_mammogram'] = True
                load_sample['mammogram'] = imgs_mammogram
            else:
                load_sample['has_mammogram'] = False
                if 'mammogram' in self.modalities:
                    load_sample['mammogram'] = torch.zeros(2, 3, self.mammogram_size[0], self.mammogram_size[1])

        if 'mri' in self.modalities:
            if sample['mri'] != None and 'mri' not in masked_modality:
                imgs_mri = self.load_mri(sample['mri'])
                imgs_mri = torch.stack(imgs_mri, dim=0)
                load_sample['has_mri'] = True
                load_sample['mri'] = imgs_mri
            else:
                load_sample['has_mri'] = False
                if 'mri' in self.modalities:
                    load_sample['mri'] = torch.zeros(self.num_mri_modalities, self.mri_size[0], 3, self.mri_size[1], self.mri_size[2])

        if 'pathology' in self.modalities:
            if sample['pathology'] != None:
                if self.mode == 'train':
                    imgs_pathology = self.load_pathology(random.choice(sample['pathology']))
                else:
                    imgs_pathology = self.load_pathology(sample['pathology'][0])
                imgs_pathology = torch.stack(imgs_pathology, dim=0)
                load_sample['has_pathology'] = True
                load_sample['pathology'] = imgs_pathology
            else:
                load_sample['has_pathology'] = False

        if self.multimodal == True:
            if 'mri' not in self.modalities:
                load_sample['has_mri'] = False
                load_sample['mri'] = torch.zeros(self.num_mri_modalities, self.mri_size[0], 3, self.mri_size[1], self.mri_size[2])
            if 'mammogram' not in self.modalities:
                load_sample['has_mammogram'] = False
                load_sample['mammogram'] = torch.zeros(2, 3, self.mammogram_size[0], self.mammogram_size[1])
            if 'ultrasound' not in self.modalities:
                load_sample['has_ultrasound'] = False
                load_sample['ultrasound'] = torch.zeros(2, 3, self.ultrasound_size[0], self.ultrasound_size[1])
        load_sample['labels'] = torch.tensor(sample['labels'])
        return load_sample

    def __len__(self):
        if self.mode == 'train':
            if self.alignment == True and self.epoch < self.epoch_alignment:
                return len(self.samples_alignment)
            else:
                return len(self.train_samples)
        if self.mode == 'validation':
            return len(self.validation_samples)
        if self.mode == 'internal':
            return len(self.internal_samples)
        if self.mode == 'external_1':
            return len(self.external_1_samples)
        if self.mode == 'external_2':
            return len(self.external_2_samples)
        if self.mode == 'reader':
            return len(self.reader_samples)

def collate_fn(batch):
    return [*batch]

def get_dataloader(batch_size, shuffle, pin_memory, num_workers, mode, opt):
    dataset = Dataset(mode, opt)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                  pin_memory=pin_memory, collate_fn=collate_fn)
    return dataset, data_loader