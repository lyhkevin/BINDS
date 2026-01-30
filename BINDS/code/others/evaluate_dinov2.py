import os
import glob
import shutil
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.dinov2 import vit_base_dinov2

PRETRAIN_PATH = "/public_bme2/Share200T/liyh2022/BINDS/weight/pretrain/pathology_dinov2/training_512499/teacher_checkpoint.pth" 
DATA_ROOT = "/public_bme2/Share200T/liyh2022/BINDS/dataset/processed/Yunnan/neoadjuvant/pathology/"
OUTPUT_DIR = "./result_neighbors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def load_dinov2(pretrain_path, device="cpu"):
    encoder = vit_base_dinov2(img_size=224, patch_size=14, block_chunks=0, init_values=1)
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location=device)
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    return encoder

class PathologyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception:
            return torch.zeros(3, 224, 224), img_path

def main():
    all_image_paths = []
    for i in range(1, 100):
        search_pattern = os.path.join(DATA_ROOT, str(i), "**", "*.png")
        files = glob.glob(search_pattern, recursive=True)
        all_image_paths.extend(files)
    
    if len(all_image_paths) == 0:
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PathologyDataset(all_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_dinov2(PRETRAIN_PATH, device=DEVICE)

    all_features = []
    valid_paths = []

    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.to(DEVICE)
            outputs = model(images) 
            outputs = F.normalize(outputs, dim=1)
            all_features.append(outputs.cpu())
            valid_paths.extend(paths)

    features_tensor = torch.cat(all_features, dim=0)
    
    query_idx = 0
    query_feature = features_tensor[query_idx].unsqueeze(0)
    query_path = valid_paths[query_idx]
    
    similarities = torch.mm(query_feature, features_tensor.t()).squeeze(0)
    topk_values, topk_indices = torch.topk(similarities, k=11)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    shutil.copy(query_path, os.path.join(OUTPUT_DIR, "0_Query_Image.png"))
    
    for rank, idx in enumerate(topk_indices[1:], start=1):
        idx = idx.item()
        score = topk_values[rank].item()
        neighbor_path = valid_paths[idx]
        filename = os.path.basename(neighbor_path)
        save_name = f"{rank}_score{score:.4f}_{filename}"
        shutil.copy(neighbor_path, os.path.join(OUTPUT_DIR, save_name))
        print(f"Rank {rank}: {save_name}")

if __name__ == "__main__":
    main()