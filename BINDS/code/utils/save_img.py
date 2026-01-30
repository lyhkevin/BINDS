import os
from torchvision.transforms.functional import to_pil_image
import imageio
from torchvision.utils import save_image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

import pandas as pd
def save_all_prediction(test_samples, preds_risk, preds_subtype, save_path):
    ids = [s[0] for s in test_samples]
    cohorts = [s[1] for s in test_samples]
    gt_risk = [s[2] for s in test_samples]
    gt_subtype = [s[3] for s in test_samples]

    data = {
        'ID': ids,
        'Cohort': cohorts,
        'GT_Risk': gt_risk,
        'GT_Subtype': gt_subtype
    }

    if isinstance(preds_risk, dict):
        for modality, preds in preds_risk.items():
            if preds is None or len(preds) == 0:
                continue
            for c in range(preds.shape[1]):
                data[f'{modality}_Prob_Risk_Class_{c}'] = preds[:, c]
    else:
        for c in range(preds_risk.shape[1]):
            data[f'Prob_Risk_Class_{c}'] = preds_risk[:, c]
    if isinstance(preds_subtype, dict):
        for modality, preds in preds_subtype.items():
            if preds is None or len(preds) == 0:
                continue
            for c in range(preds.shape[1]):
                data[f'{modality}_Prob_Subtype_Class_{c}'] = preds[:, c]
    else:
        for c in range(preds_subtype.shape[1]):
            data[f'Prob_Subtype_Class_{c}'] = preds_subtype[:, c]
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Cohort', 'ID']).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        df.to_excel(save_path, index=False)
    except Exception:
        df.to_csv(save_path.replace('.xlsx', '.csv'), index=False)

def apply_colormap(attention_map):
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    colormap = plt.get_cmap('jet')
    attention_map_rgb = colormap(attention_map)
    attention_map_rgb = attention_map_rgb[..., :3]
    return attention_map_rgb

def save_snapshot_with_attention(opt, save_path, sample, target, attention_ultrasound, attention_mammogram, attention_mri):
    id, cohort, labels = sample['id'], sample['cohort'], sample['labels']
    save_path = save_path + sample['cohort'] + '_' + str(id)
    for i in range(len(target)):
        save_path = save_path + '_' + target[i] + '_' + str(labels[i].item())
    os.makedirs(save_path, exist_ok=True)
    if sample['has_ultrasound'] == True:
        os.makedirs(save_path + '/ultrasound/', exist_ok=True)
        save_us(sample['ultrasound'], save_path + '/ultrasound/')
        save_us_attention(sample['ultrasound'], attention_ultrasound, save_path + '/ultrasound/')
    if sample['has_mammogram'] == True:
        os.makedirs(save_path + '/mammogram/', exist_ok=True)
        save_x(sample['mammogram'], save_path + '/mammogram/')
        save_x_attention(sample['mammogram'], attention_mammogram, save_path + '/mammogram/')
    if sample['has_mri'] == True:
        os.makedirs(save_path + '/mri/', exist_ok=True)
        save_mri_attention(sample['mri'], attention_mri, save_path + '/mri/')
    if sample['has_pathology'] == True:
        os.makedirs(save_path + '/pathology/', exist_ok=True)
        save_pathology(sample['pathology'], save_path + '/pathology/')
    return

def save_predictions(subject_id, sample, target, save_path, predictions):
    id, cohort, labels = sample['id'], sample['cohort'], sample['labels']
    save_path = save_path + sample['cohort'] + '_' + str(id)
    for i in range(len(target)):
        save_path = save_path + '_' + target[i] + '_' + str(labels[i].item())
    prediction_file = os.path.join(save_path, 'predictions.txt')
    with open(prediction_file, 'a') as file:
        for modality, prediction in predictions.items():
            prediction_risk, prediction_subtype = prediction[0], prediction[1]
            if isinstance(prediction_risk, torch.Tensor):
                risk_value = prediction_risk[subject_id].tolist()
            else:
                risk_value = prediction_risk[subject_id]
            if isinstance(prediction_subtype, torch.Tensor):
                subtype_value = prediction_subtype[subject_id].tolist()
            else:
                subtype_value = prediction_subtype[subject_id]
            file.write(f"Modality: {modality}\n")
            file.write(f"Risk Prediction: {risk_value}\n")
            file.write(f"Subtype Prediction: {subtype_value}\n")
            file.write("\n")

def save_contribution_scores(subject_id, sample, target, save_path, contribution, modality):
    id, cohort, labels = sample['id'], sample['cohort'], sample['labels']
    save_path = save_path + sample['cohort'] + '_' + str(id)
    for i in range(len(target)):
        save_path = save_path + '_' + target[i] + '_' + str(labels[i].item())
    prediction_file = os.path.join(save_path, 'contribution.txt')
    
    if modality == "ultrasound":
        explanation = "Contribution scores for different ultrasound input views [transverse view, longitudinal view]"
    if modality == "mammogram":
        explanation = "Contribution scores for different mammogram input views [cc view, mlo view]"
    if modality == "mri":
        explanation = "Contribution scores for different mri input modalities [ADC, P0, P2, T2]"
    if modality == "bimodal":
        explanation = "Contribution scores for different input modalities [ultrasound, mammogram]"
    if modality == "multimodal":
        explanation = "Contribution scores for different input modalities [ultrasound, mammogram, mri]"
    
    with open(prediction_file, 'a') as file:
        file.write(explanation + "\n")
        if isinstance(contribution, torch.Tensor):
            file.write(f"Contribution: {contribution[subject_id].tolist()}\n")
        else:
            file.write(f"Contribution: {contribution[subject_id]}\n")
        file.write("\n")

def keep_top_percent(attn_map, ratio):
    threshold = torch.quantile(attn_map, 1 - ratio)
    mask = (attn_map >= threshold).float()
    return attn_map * mask

def save_us_attention(imgs_ultrasound, attention_ultrasound, save_path):
    img_view_1, img_view_2 = imgs_ultrasound[0], imgs_ultrasound[1]
    attn = attention_ultrasound.mean(dim=0)
    attn = attn.reshape(2, attn.shape[0] // 2)

    attn_view_1, attn_view_2 = attn[0], attn[1]
    w_featmap = int(attn_view_1.size(-1) ** 0.5)
    scale_factor = int(224 / w_featmap)
    attn_view_1 = attn_view_1.reshape(1, w_featmap, w_featmap)
    attn_view_2 = attn_view_2.reshape(1, w_featmap, w_featmap)
    attn_view_1 = nn.functional.interpolate(attn_view_1.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0].cpu().detach().numpy()
    attn_view_2 = nn.functional.interpolate(attn_view_2.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0].cpu().detach().numpy()
    attn_view_1 = apply_colormap(attn_view_1[0])
    attn_view_2 = apply_colormap(attn_view_2[0])
    img_view_1_np = img_view_1.cpu().detach().numpy().transpose(1, 2, 0)
    img_view_2_np = img_view_2.cpu().detach().numpy().transpose(1, 2, 0)
    overlay_view_1 = 0.8 * img_view_1_np + 0.2 * attn_view_1
    overlay_view_2 = 0.8 * img_view_2_np + 0.2 * attn_view_2
    overlay_view_1 = np.clip(overlay_view_1, 0, 1)
    overlay_view_2 = np.clip(overlay_view_2, 0, 1)
    plt.imsave(fname=save_path + f'/us_attn_0.png', arr=overlay_view_1, format='png')
    plt.imsave(fname=save_path + f'/us_attn_1.png', arr=overlay_view_2, format='png')

def save_x_attention(imgs_mammogram, attention_mammogram, save_path):
    img_view_1, img_view_2 = imgs_mammogram[0], imgs_mammogram[1]
    attn = attention_mammogram.mean(dim=0)
    attn = attn.reshape(2, attn.shape[0] // 2)
    attn_view_1, attn_view_2 = attn[0], attn[1]

    w_featmap = int(attn_view_1.size(-1) ** 0.5)
    scale_factor = int(224 / w_featmap)
    attn_view_1 = attn_view_1.reshape(1, w_featmap, w_featmap)
    attn_view_2 = attn_view_2.reshape(1, w_featmap, w_featmap)
    attn_view_1 = nn.functional.interpolate(attn_view_1.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0].cpu().detach().numpy()
    attn_view_2 = nn.functional.interpolate(attn_view_2.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0].cpu().detach().numpy()
    attn_view_1 = apply_colormap(attn_view_1[0])
    attn_view_2 = apply_colormap(attn_view_2[0])
    img_view_1_np = img_view_1.cpu().detach().numpy().transpose(1, 2, 0)
    img_view_2_np = img_view_2.cpu().detach().numpy().transpose(1, 2, 0)
    overlay_view_1 = 0.8 * img_view_1_np + 0.2 * attn_view_1
    overlay_view_2 = 0.8 * img_view_2_np + 0.2 * attn_view_2
    overlay_view_1 = np.clip(overlay_view_1, 0, 1)
    overlay_view_2 = np.clip(overlay_view_2, 0, 1)
    plt.imsave(fname=save_path + f'/x_attn_0.png', arr=overlay_view_1, format='png')
    plt.imsave(fname=save_path + f'/x_attn_1.png', arr=overlay_view_2, format='png')

def save_mri_attention(imgs_mri, attention_mri, save_path):
    img_list = [imgs_mri[0], imgs_mri[1], imgs_mri[2], imgs_mri[3]]
    modality_mapping = ['ADC', 'P0', 'P2', 'T2']
    attn = attention_mri.mean(dim=0)
    attn = attn.reshape(4, attn.shape[0] // 4)
    for j in range(4):
        attn_mri = attn[j]
        w_featmap = int(attn.size(-1) ** 0.5)
        attn_mri = attn_mri.reshape(1, w_featmap, w_featmap)
        scale_factor = int(96 / w_featmap)
        attn_mri = nn.functional.interpolate(attn_mri.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0].cpu().detach().numpy()
        modality_save_path = os.path.join(save_path, modality_mapping[j])
        os.makedirs(modality_save_path, exist_ok=True)
        num_slices = img_list[j].size(0)
        for s in range(num_slices):
            if s % 20 != 0:
                continue
            img_np = img_list[j][s].cpu().detach().numpy().transpose(1, 2, 0)
            heatmap = apply_colormap(attn_mri[0])
            overlay = 0.8 * img_np + 0.2 * heatmap
            overlay = np.clip(overlay, 0, 1)
            plt.imsave(fname=os.path.join(modality_save_path, f'mri_attn_slice_{s}.png'), arr=overlay, format='png')
        for s in range(num_slices):
            if s % 20 != 0:
                continue
            save_image(img_list[j][s], os.path.join(modality_save_path, f'mri_slice_{s}.png'))

def save_snapshot(opt, save_path, sample, target, prediction = None):
    id, cohort, labels = sample['id'], sample['cohort'], sample['labels']
    save_path = save_path + sample['cohort'] + '_' + str(id)
    for i in range(len(target)):
        save_path = save_path + '_' + target[i] + '_' + str(labels[i].item())
    os.makedirs(save_path, exist_ok=True)
    if sample['ultrasound'] is not None:
        save_us(sample['ultrasound'], save_path)
    if sample['mammogram'] is not None:
        save_x(sample['mammogram'], save_path)
    if sample['mri'] is not None:
        save_mri(opt, sample['mri'], save_path)
    if sample['pathology'] is not None:
        save_pathology(sample['pathology'], save_path)
    return

def save_mri(opt, radiology, save_path):
    images = {modality: [] for modality in opt.mri_modalities}
    for i in range(radiology.shape[1]):
        for index, modality in enumerate(opt.mri_modalities):
            pil_image = to_pil_image(radiology[index][i])
            images[modality].append(pil_image)
    for modality, imgs in images.items():
        imageio.mimsave(f'{save_path}/{modality}.gif', imgs, duration=0.1)

def save_us(radiology, save_path):
    save_image(radiology[0], save_path + '/us_view_0.png')
    save_image(radiology[1], save_path + '/us_view_1.png')

def save_x(radiology, save_path):
    save_image(radiology[0], save_path + '/cc_view.png')
    save_image(radiology[1], save_path + '/mlo_view.png')

def save_pathology(pathology, save_path):
    save_image(pathology, save_path + '/pathology.png', nrow = 10)