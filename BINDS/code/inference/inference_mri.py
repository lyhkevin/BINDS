import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mri_encoder import *
from utils.dataloader import *
from utils.test import *
from utils.save_img import *
from utils.option import *
from tqdm import tqdm

if __name__ == '__main__':
    opt = Options().get_opt()
    opt.batch_size = 10
    opt.num_workers = 4
    opt.modalities = ['mri']
    opt.key_modalities = ['mri']
    opt.save_path = '../../evaluation/mri/'
    opt.weight_save_path_mri = '../weight/mri/best_encoder_mri.pth'
    opt.radiological = True
    opt.augmentation = False
    opt.target = ['Risk', 'Subtype']
    opt.oversample = False
    opt.split_data = False
    opt.multimodal = False
    opt.data_split_path = '../dataset/data.json'
    opt.save_snapshot = False

    for key in opt.__dict__:
        if isinstance(opt.__dict__[key], str):
            if 'weight' in opt.__dict__[key] or 'dataset' in opt.__dict__[key]:
                opt.__dict__[key] = '../' + opt.__dict__[key]

    os.makedirs(opt.save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset, data_loader = get_dataloader(batch_size=opt.batch_size, shuffle=False, pin_memory=False, num_workers=opt.num_workers, mode='internal', opt=opt)
    num_classes = dataset.num_classes
    
    encoder_mri = mri_encoder(
        modalities=opt.mri_modalities, embed_dim=opt.embed_dim, backbone=opt.backbone_mri,
        num_classes=num_classes, depth_encoder=opt.depth_encoder_mri, depth_fusion=opt.depth_fusion_mri, multi_modal=True).to(device)
    
    encoder_mri.load_state_dict(torch.load(opt.weight_save_path_mri), strict=True)
    encoder_mri.eval()

    preds_risk = {'ultrasound': [], 'mammogram': [], 'mri': []}
    preds_subtype = {'ultrasound': [], 'mammogram': [], 'mri': []}
    gts = {'risk': [], 'subtype': []}
    features = {'risk': [], 'subtype': []}
    test_samples = []
    
    for i, samples in enumerate(tqdm(data_loader)):

        imgs_mri = []
        has_mri = []
        labels = []

        for sample in samples:
            has_mri.append(sample['has_mri'])
            imgs_mri.append(sample['mri'])
            labels.append(sample['labels'])
            test_samples.append((sample['id'], sample['cohort'], sample['labels'][0].item(), sample['labels'][1].item()))

        imgs_mri = torch.stack(imgs_mri, dim=0).to(device)
        labels = torch.stack(labels, dim=1).to(device)

        encoder_mri.zero_grad()
        f_mri, predictions_mri, attention_mri, importance_mri = encoder_mri.forward_with_grad(imgs_mri)
        predictions_mri[0], predictions_mri[1] = F.softmax(predictions_mri[0], dim=-1).detach().cpu(), F.softmax(predictions_mri[1], dim=-1).detach().cpu()
        
        for j in range(f_mri.shape[0]):
            features['risk'].append(f_mri[j][0].detach().cpu())
            features['subtype'].append(f_mri[j][1].detach().cpu())
        
        preds_risk['mri'].append(predictions_mri[0])
        preds_subtype['mri'].append(predictions_mri[1])
        gts['risk'].append(labels[0].cpu().numpy())
        gts['subtype'].append(labels[1].cpu().numpy())
        
        predictions = {'mri': predictions_mri}

        for j, sample in enumerate(samples):
            save_snapshot_with_attention(opt, opt.save_path + '/attention_maps/', sample, opt.target, None, None, attention_mri[-1][j][:, 0, 2:])
            save_predictions(j, sample, opt.target, opt.save_path + '/attention_maps/', predictions)
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_mri, 'mri')

    for modality in preds_risk.keys():
        if len(preds_risk[modality]) > 0:
            preds_risk[modality] = torch.cat(preds_risk[modality], dim=0).cpu().numpy()
            preds_subtype[modality] = torch.cat(preds_subtype[modality], dim=0).cpu().numpy()

    for key in gts.keys():
        if gts[key] and len(gts[key]) > 0:
            gts[key] = np.concatenate(gts[key], axis=0)
            
    features['risk'] = torch.stack(features['risk'], dim=0).cpu().numpy()
    features['subtype'] = torch.stack(features['subtype'], dim=0).cpu().numpy()

    save_all_prediction(test_samples, preds_risk, preds_subtype, opt.save_path + 'mri_predictions.xlsx')

    np.savez(
        opt.save_path + 'mri_predictions.npz',
        preds_risk = preds_risk,
        preds_subtype = preds_subtype,
        features = features,
        gts=gts
    )