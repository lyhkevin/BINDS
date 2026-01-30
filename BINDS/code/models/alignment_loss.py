import torch
import torch.nn.functional as F

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
def feature_alignment(features_radiology, features_pathology, has_img, has_pathology, num_layers = 3):
    loss = 0
    num_paired = 0
    features_radiology, features_pathology = features_radiology[-num_layers:], features_pathology[-num_layers:]
    for i, (img, pathology) in enumerate(zip(has_img, has_pathology)):
        if img and pathology:
            loss_alignment = 0
            for feature_radiology, feature_pathology in zip(features_radiology, features_pathology):
                loss_alignment = loss_alignment + sce_loss(feature_radiology[i], feature_pathology[num_paired])
            loss_alignment = loss_alignment / num_layers
            loss += loss_alignment
            num_paired += 1
    if num_paired == 0:
        return False
    loss = loss / num_paired
    return loss

