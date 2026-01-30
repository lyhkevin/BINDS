import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ultrasound_encoder import *
from models.mammogram_encoder import *
from models.mri_encoder import *
from models.multimodal_predictor import *
from utils.dataloader import *
from utils.test import *
from utils.save_img import *
from utils.option import *
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    
    opt = Options().get_opt()
    opt.batch_size = 3
    opt.num_workers = 4
    opt.depth_fusion = 6
    opt.modalities = ['ultrasound', 'mammogram', 'mri']
    opt.key_modalities = []
    opt.min_num_modalities = 2
    opt.save_path = '../../evaluation/two_stage/'
    
    opt.weight_save_path_ultrasound = '../weight/ultrasound/best_encoder_ultrasound.pth'
    opt.weight_save_path_mammogram = '../weight/mammogram/best_encoder_mammogram.pth'
    opt.weight_save_path_mri = '../weight/mri/best_encoder_mri.pth'
    opt.weight_save_path_predictor = '../weight/multi_modal/best_predictor.pth'
    
    opt.radiological = True
    opt.augmentation = False
    opt.multimodal = True
    opt.target = ['Risk', 'Subtype']
    opt.oversample = False
    opt.split_data = False
    opt.data_split_path = '../dataset/data.json'
    opt.save_snapshot = False
    
    opt.threshold_low = 0.02
    opt.threshold_high = 0.98

    for key in opt.__dict__:
        if isinstance(opt.__dict__[key], str):
            if 'weight' in opt.__dict__[key] or 'dataset' in opt.__dict__[key]:
                opt.__dict__[key] = '../' + opt.__dict__[key]

    os.makedirs(opt.save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    internal_dataset, internal_loader = get_dataloader(batch_size=opt.batch_size, shuffle=False, pin_memory=False,
                                                       num_workers=opt.num_workers, mode='internal', opt=opt)
    num_classes = internal_dataset.num_classes

    encoder_ultrasound = ultrasound_encoder(embed_dim=opt.embed_dim, backbone=opt.backbone_ultrasound,
                                            num_classes=num_classes, depth_encoder=opt.depth_encoder_ultrasound,
                                            depth_fusion=opt.depth_fusion_ultrasound, multi_modal=True, dropout=0).to(device)
    
    encoder_mammogram = mammogram_encoder(embed_dim=opt.embed_dim, backbone=opt.backbone_mammogram,
                                          depth_encoder=opt.depth_encoder_mammogram, num_classes=num_classes,
                                          depth_fusion=opt.depth_fusion_mammogram, multi_modal=True, dropout=0).to(device)

    encoder_mri = mri_encoder(modalities=opt.mri_modalities, embed_dim=opt.embed_dim, backbone=opt.backbone_mri,
                              num_classes=num_classes, depth_encoder=opt.depth_encoder_mri, 
                              depth_fusion=opt.depth_fusion_mri, multi_modal=True).to(device)
    
    predictor = multimodal_predictor(device=device, num_heads=opt.num_heads, num_classes=num_classes,
                                     embed_dim=opt.embed_dim, depth=opt.depth_fusion, dropout=0).to(device)

    encoder_mammogram.load_state_dict(torch.load(opt.weight_save_path_mammogram), strict=True)
    encoder_ultrasound.load_state_dict(torch.load(opt.weight_save_path_ultrasound), strict=True)
    encoder_mri.load_state_dict(torch.load(opt.weight_save_path_mri), strict=True)
    predictor.load_state_dict(torch.load(opt.weight_save_path_predictor), strict=True)

    encoder_ultrasound.eval()
    encoder_mammogram.eval()
    encoder_mri.eval()
    predictor.eval()

    preds_risk = {'ultrasound': [], 'mammogram': [], 'mri': [], 'initial': [], 'final': []}
    preds_subtype = {'ultrasound': [], 'mammogram': [], 'mri': [], 'initial': [], 'final': []}
    
    gts = {'risk': [], 'subtype': []}
    features = {'risk': [], 'subtype': []}
    test_samples = []
    
    for i, samples in enumerate(tqdm(internal_loader)):
        
        imgs_ultrasound = []
        imgs_mammogram = []
        imgs_mri = []
        has_ultrasound = []
        has_mammogram = []
        has_mri = []
        labels = []
        
        for sample in samples:
            has_ultrasound.append(sample['has_ultrasound'])
            has_mammogram.append(sample['has_mammogram'])
            has_mri.append(sample['has_mri'])
            imgs_ultrasound.append(sample['ultrasound'])
            imgs_mammogram.append(sample['mammogram'])
            imgs_mri.append(sample['mri'])
            labels.append(sample['labels'])
            test_samples.append((sample['id'], sample['cohort'], sample['labels'][0].item(), sample['labels'][1].item()))

        imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0).to(device)
        imgs_mammogram = torch.stack(imgs_mammogram, dim=0).to(device)
        imgs_mri = torch.stack(imgs_mri, dim=0).to(device)
        labels = torch.stack(labels, dim=1).to(device)

        f_ultrasound, predictions_ultrasound, attention_ultrasound, importance_ultrasound = encoder_ultrasound.forward_with_grad(imgs_ultrasound)
        f_mammogram, predictions_mammogram, attention_mammogram, importance_mammogram = encoder_mammogram.forward_with_grad(imgs_mammogram)
        f_mri, predictions_mri, attention_mri, importance_mri = encoder_mri.forward_with_grad(imgs_mri)
        
        f_multimodal, predictions_multimodal, attention_multimodal, importance_multimodal = predictor.forward_with_grad(
            has_ultrasound, has_mammogram, has_mri, f_ultrasound.detach(), f_mammogram.detach(), f_mri.detach())
        
        for j in range(f_multimodal.shape[0]):
            features['risk'].append(f_multimodal[j][0].detach().cpu())
            features['subtype'].append(f_multimodal[j][1].detach().cpu())
            
        for j in range(len(has_mri)):
            has_mri[j] = False
        f_bimodal, predictions_bimodal, attention_bimodal, importance_bimodal = predictor.forward_with_grad(
            has_ultrasound, has_mammogram, has_mri, f_ultrasound.detach(), f_mammogram.detach(), f_mri.detach())
        
        predictions_ultrasound[0], predictions_ultrasound[1] = F.softmax(predictions_ultrasound[0], dim=-1).detach().cpu(), F.softmax(predictions_ultrasound[1], dim=-1).detach().cpu()
        predictions_mammogram[0], predictions_mammogram[1] = F.softmax(predictions_mammogram[0], dim=-1).detach().cpu(), F.softmax(predictions_mammogram[1], dim=-1).detach().cpu()
        predictions_mri[0], predictions_mri[1] = F.softmax(predictions_mri[0], dim=-1).detach().cpu(), F.softmax(predictions_mri[1], dim=-1).detach().cpu()
        predictions_bimodal[0], predictions_bimodal[1] = F.softmax(predictions_bimodal[0], dim=-1).detach().cpu(), F.softmax(predictions_bimodal[1], dim=-1).detach().cpu()
        predictions_multimodal[0], predictions_multimodal[1] = F.softmax(predictions_multimodal[0], dim=-1).detach().cpu(), F.softmax(predictions_multimodal[1], dim=-1).detach().cpu()

        stage1_risk = []
        stage1_subtype = []
        has_mri = [sample['has_mri'] for sample in samples]
        for j in range(len(samples)):
            if has_ultrasound[j] and has_mammogram[j]:
                stage1_risk.append(predictions_bimodal[0][j])
                stage1_subtype.append(predictions_bimodal[1][j])
            elif has_ultrasound[j]:
                stage1_risk.append(predictions_ultrasound[0][j])
                stage1_subtype.append(predictions_ultrasound[1][j])
            elif has_mammogram[j]:
                stage1_risk.append(predictions_mammogram[0][j])
                stage1_subtype.append(predictions_mammogram[1][j])
        predictions_initial = (torch.stack(stage1_risk), torch.stack(stage1_subtype))

        final_risk = []
        final_subtype = []
        for j in range(len(samples)):
            prob_risk_initial = predictions_initial[0][j][1].item()
            is_uncertain = opt.threshold_low < prob_risk_initial < opt.threshold_high
            if is_uncertain and has_mri[j]:
                final_risk.append(predictions_multimodal[0][j])
                final_subtype.append(predictions_multimodal[1][j])
            else:
                final_risk.append(predictions_initial[0][j])
                final_subtype.append(predictions_initial[1][j])
        predictions_final = (torch.stack(final_risk), torch.stack(final_subtype))
    
        preds_risk['ultrasound'].append(predictions_ultrasound[0])
        preds_risk['mammogram'].append(predictions_mammogram[0])
        preds_risk['mri'].append(predictions_mri[0])
        preds_risk['initial'].append(predictions_initial[0])
        preds_risk['final'].append(predictions_final[0])

        preds_subtype['ultrasound'].append(predictions_ultrasound[1])
        preds_subtype['mammogram'].append(predictions_mammogram[1])
        preds_subtype['mri'].append(predictions_mri[1])
        preds_subtype['initial'].append(predictions_initial[1])
        preds_subtype['final'].append(predictions_final[1])

        gts['risk'].append(labels[0].cpu().numpy())
        gts['subtype'].append(labels[1].cpu().numpy())

        predictions = {
            'ultrasound': predictions_ultrasound,
            'mammogram': predictions_mammogram,
            'mri': predictions_mri,
            'initial': predictions_initial,
            'final': predictions_final,
        }

        for j, sample in enumerate(samples):
            save_snapshot_with_attention(opt, opt.save_path + '/attention_maps/', sample, opt.target, 
                                         attention_ultrasound[-1][j][:, 0, 2:], 
                                         attention_mammogram[-1][j][:, 0, 2:], 
                                         attention_mri[-1][j][:, 0, 2:])
            save_predictions(j, sample, opt.target, opt.save_path + '/attention_maps/', predictions)
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_ultrasound, 'ultrasound')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_mammogram, 'mammogram')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_mri, 'mri')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_bimodal, 'bimodal')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_multimodal, 'multimodal')

    for modality in preds_risk.keys():
        if len(preds_risk[modality]) > 0:
            preds_risk[modality] = torch.cat(preds_risk[modality], dim=0).cpu().numpy()
            preds_subtype[modality] = torch.cat(preds_subtype[modality], dim=0).cpu().numpy()

    for key in gts.keys():
        if gts[key] and len(gts[key]) > 0:
            gts[key] = np.concatenate(gts[key], axis=0)

    features['risk'] = torch.stack(features['risk'], dim=0).cpu().numpy()
    features['subtype'] = torch.stack(features['subtype'], dim=0).cpu().numpy()
    save_all_prediction(test_samples, preds_risk, preds_subtype, opt.save_path + 'two_stage_predictions.xlsx')

    np.savez(
        opt.save_path + 'two_stage_predictions.npz',
        preds_risk = preds_risk,
        preds_subtype = preds_subtype,
        features = features,
        gts = gts,
        ids = [t[0] for t in test_samples]
    )