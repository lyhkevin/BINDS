from glob import glob
import os
import openslide
from tqdm import tqdm

class dataset_info:
    def __init__(self):
        self.svs_path = glob('/path/', recursive=True)
        for i in range(len(self.svs_path)):
            self.svs_path[i] = self.svs_path[i].replace("\\", "/")
        self.base_dir = '/path/'
        os.makedirs(self.base_dir, exist_ok=True)
        self.svs = None
        self.slide = None
        self.svs_name = None
        self.w_1x = None
        self.h_1x = None
        self.w_40x = None
        self.h_40x = None

    def makedir(self):
        self.id = self.path.split('/')[-2]
        self.name = self.path.split('/')[-1][:-4]

def read_wsi(info):
    print("load svs...........")
    print('svs path:', info.path)
    info.slide = openslide.OpenSlide(info.path)
    # info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    # thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    thumbnail = info.slide.get_thumbnail((2048, 2048))
    thumbnail.save(info.base_dir + info.name + '.png')

if __name__ == '__main__':
    info = dataset_info()
    for svs_path in tqdm(info.svs_path):
        info.path = svs_path
        num_patches = info.makedir()
        print('preprocessing whole slide image:', svs_path)
        read_wsi(info)