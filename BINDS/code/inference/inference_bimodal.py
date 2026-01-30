import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ultrasound_encoder import *
from models.mammogram_encoder import *
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
    opt.modalities = ['ultrasound', 'mammogram']
    opt.key_modalities = ['ultrasound', 'mammogram']
    opt.min_num_modalities = 2
    opt.save_path = '../../evaluation/bimodal/'
    
    opt.weight_save_path_ultrasound = '../weight/ultrasound/best_encoder_ultrasound.pth'
    opt.weight_save_path_mammogram = '../weight/mammogram/best_encoder_mammogram.pth'
    opt.weight_save_path_predictor = '../weight/multi_modal/best_predictor.pth'
    
    opt.radiological = True
    opt.augmentation = False
    opt.multimodal = True
    opt.target = ['Risk', 'Subtype']
    opt.oversample = False
    opt.split_data = False
    opt.data_split_path = '../dataset/data.json'
    opt.save_snapshot = False

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
    predictor = multimodal_predictor(device=device, num_heads=opt.num_heads, num_classes=num_classes,
                                     embed_dim=opt.embed_dim, depth=opt.depth_fusion, dropout=0).to(device)

    encoder_mammogram.load_state_dict(torch.load(opt.weight_save_path_mammogram), strict=True)
    encoder_ultrasound.load_state_dict(torch.load(opt.weight_save_path_ultrasound), strict=True)
    predictor.load_state_dict(torch.load(opt.weight_save_path_predictor), strict=True)

    encoder_ultrasound.eval()
    encoder_mammogram.eval()
    predictor.eval()

    preds_risk = {'ultrasound': [], 'mammogram': [], 'bimodal': []}
    preds_subtype = {'ultrasound': [], 'mammogram': [], 'bimodal': []}
    
    gts = {'risk': [], 'subtype': []}
    features = {'risk': [], 'subtype': []}
    test_samples = []

    for i, samples in enumerate(tqdm(internal_loader)):
        
        imgs_ultrasound = []
        imgs_mammogram = []
        has_ultrasound = []
        has_mammogram = []
        has_mri = []
        labels = []
        
        for sample in samples:
            has_mri.append(False)
            has_ultrasound.append(sample['has_ultrasound'])
            has_mammogram.append(sample['has_mammogram'])
            imgs_ultrasound.append(sample['ultrasound'])
            imgs_mammogram.append(sample['mammogram'])
            labels.append(sample['labels'])
            test_samples.append((sample['id'], sample['cohort'], sample['labels'][0].item(), sample['labels'][1].item()))

        imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0).to(device)
        imgs_mammogram = torch.stack(imgs_mammogram, dim=0).to(device)
        labels = torch.stack(labels, dim=1).to(device)

        f_ultrasound, predictions_ultrasound, attention_ultrasound, importance_ultrasound = encoder_ultrasound.forward_with_grad(imgs_ultrasound)
        f_mammogram, predictions_mammogram, attention_mammogram, importance_mammogram = encoder_mammogram.forward_with_grad(imgs_mammogram)

        f_bimodal, predictions_bimodal, attention_bimodal, importance_bimodal = predictor.forward_with_grad(
            has_ultrasound, has_mammogram, has_mri, 
            f_ultrasound.detach(), f_mammogram.detach(), None
        )

        predictions_ultrasound[0], predictions_ultrasound[1] = F.softmax(predictions_ultrasound[0], dim=-1).detach().cpu(), F.softmax(predictions_ultrasound[1], dim=-1).detach().cpu()
        predictions_mammogram[0], predictions_mammogram[1] = F.softmax(predictions_mammogram[0], dim=-1).detach().cpu(), F.softmax(predictions_mammogram[1], dim=-1).detach().cpu()
        predictions_bimodal[0], predictions_bimodal[1] = F.softmax(predictions_bimodal[0], dim=-1).detach().cpu(), F.softmax(predictions_bimodal[1], dim=-1).detach().cpu()

        for j in range(f_bimodal.shape[0]):
            features['risk'].append(f_bimodal[j][0].detach().cpu())
            features['subtype'].append(f_bimodal[j][1].detach().cpu())

        preds_risk['ultrasound'].append(predictions_ultrasound[0])
        preds_risk['mammogram'].append(predictions_mammogram[0])
        preds_risk['bimodal'].append(predictions_bimodal[0])

        preds_subtype['ultrasound'].append(predictions_ultrasound[1])
        preds_subtype['mammogram'].append(predictions_mammogram[1])
        preds_subtype['bimodal'].append(predictions_bimodal[1])
        
        gts['risk'].append(labels[0].cpu().numpy())
        gts['subtype'].append(labels[1].cpu().numpy())

        predictions = {
            'ultrasound': predictions_ultrasound,
            'mammogram': predictions_mammogram,
            'bimodal': predictions_bimodal
        }

        for j, sample in enumerate(samples):
            save_snapshot_with_attention(opt, opt.save_path + '/attention_maps/', sample, opt.target, 
                                         attention_ultrasound[-1][j][:, 0, 2:], 
                                         attention_mammogram[-1][j][:, 0, 2:], None)
            save_predictions(j, sample, opt.target, opt.save_path + '/attention_maps/', predictions)
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_ultrasound, 'ultrasound')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_mammogram, 'mammogram')
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_bimodal, 'bimodal')

    for modality in preds_risk.keys():
        if len(preds_risk[modality]) > 0:
            preds_risk[modality] = torch.cat(preds_risk[modality], dim=0).cpu().numpy()
            preds_subtype[modality] = torch.cat(preds_subtype[modality], dim=0).cpu().numpy()

    for key in gts.keys():
        if gts[key] and len(gts[key]) > 0:
            gts[key] = np.concatenate(gts[key], axis=0)

    features['risk'] = torch.stack(features['risk'], dim=0).cpu().numpy()
    features['subtype'] = torch.stack(features['subtype'], dim=0).cpu().numpy()

    save_all_prediction(test_samples, preds_risk, preds_subtype, opt.save_path + 'bimodal_predictions.xlsx')

    np.savez(
        opt.save_path + 'bimodal_predictions.npz',
        preds_risk = preds_risk,
        preds_subtype = preds_subtype,
        features = features,
        gts = gts,
        ids = [t[0] for t in test_samples]
    )
    