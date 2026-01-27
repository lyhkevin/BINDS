import os
import shutil
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from glob import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader    
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from model.dinov2 import vit_base_dinov2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
DINO_CHECKPOINT = '../example_data_and_weight/pathology/weight/dinov2.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dinov2(pretrain_path, device):
    model = vit_base_dinov2(img_size=224, patch_size=14, block_chunks=0, init_values=1)
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

class ImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_list[idx]).convert('RGB')
            image = self.transform(image)
            return image
        except Exception:
            return torch.zeros(3, 224, 224)

if __name__ == '__main__':
    num_clusters = 10
    scales = ['small', 'medium', 'large']

    model = load_dinov2(DINO_CHECKPOINT, DEVICE)
    
    slide_ids = glob('./patches/*')
    
    for slide_id in tqdm(slide_ids):
        slide_id = slide_id.replace('\\', '/')
        slide_name = slide_id.split('/')[-1]
        
        for scale in scales:
            patch_path = glob(slide_id + '/' + scale + '/*.png')
            
            if len(patch_path) == 0:
                continue
                
            current_k = num_clusters if len(patch_path) >= 10 else len(patch_path)
            
            subject_path = './representative_patches/' + slide_name + '/' + scale + '/'
            
            if len(glob(subject_path + '*.png')) > 0:
                print('already exist')
                continue

            dataset = ImageDataset(patch_path)
            dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
            
            features_list = []
            with torch.no_grad():
                for images in dataloader:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    outputs = F.normalize(outputs, dim=1)
                    features_list.append(outputs.cpu().numpy())
            
            if len(features_list) == 0:
                continue

            features = np.concatenate(features_list, axis=0)

            if features.shape[0] < current_k:
                current_k = features.shape[0]

            kmeans = KMeans(n_clusters=current_k, n_init=10, random_state=0)
            labels = kmeans.fit_predict(features)
            cluster_centers = kmeans.cluster_centers_

            distances = cdist(features, cluster_centers, 'euclidean')
            closest_indices = distances.argsort(axis=0)
            
            os.makedirs(subject_path, exist_ok=True)
            
            for j in range(current_k):
                index = closest_indices[0, j]
                original_file = patch_path[index]
                shutil.copy(original_file, os.path.join(subject_path, f"{j}.png"))