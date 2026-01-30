import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ultrasound_encoder import *
from utils.dataloader import *
from utils.test import *
from utils.save_img import *
from utils.option import *
from tqdm import tqdm

if __name__ == '__main__':
    opt = Options().get_opt()
    opt.batch_size = 50
    opt.num_workers = 4
    opt.modalities = ['ultrasound']
    opt.key_modalities = ['ultrasound']
    opt.save_path = '../../evaluation/ultrasound/'
    opt.weight_save_path_ultrasound = '../weight/ultrasound/best_encoder_ultrasound.pth'
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

    encoder_ultrasound = ultrasound_encoder(embed_dim=opt.embed_dim, backbone=opt.backbone_ultrasound,
                                            num_classes=num_classes, depth_encoder=opt.depth_encoder_ultrasound,
                                            depth_fusion=opt.depth_fusion_ultrasound, multi_modal=True).to(device)
    encoder_ultrasound.load_state_dict(torch.load(opt.weight_save_path_ultrasound), strict=True)
    encoder_ultrasound.eval()

    preds_risk = {'ultrasound': [], 'mammogram': [], 'mri': []}
    preds_subtype = {'ultrasound': [], 'mammogram': [], 'mri': []}
    gts = {'risk': [], 'subtype': []}
    features = {'risk': [], 'subtype': []}
    test_samples = []
    
    for i, samples in enumerate(tqdm(data_loader)):

        imgs_ultrasound = []
        has_ultrasound = []
        labels = []

        for sample in samples:
            has_ultrasound.append(sample['has_ultrasound'])
            imgs_ultrasound.append(sample['ultrasound'])
            labels.append(sample['labels'])
            test_samples.append((sample['id'], sample['cohort'], sample['labels'][0].item(), sample['labels'][1].item()))

        imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0).to(device)
        labels = torch.stack(labels, dim=1).to(device)

        encoder_ultrasound.zero_grad()
        f_ultrasound, predictions_ultrasound, attention_ultrasound, importance_ultrasound = encoder_ultrasound.forward_with_grad(imgs_ultrasound)
        predictions_ultrasound[0], predictions_ultrasound[1] = F.softmax(predictions_ultrasound[0], dim=-1).detach().cpu(), F.softmax(predictions_ultrasound[1], dim=-1).detach().cpu()
        
        for j in range(f_ultrasound.shape[0]):
            features['risk'].append(f_ultrasound[j][0].detach().cpu())
            features['subtype'].append(f_ultrasound[j][1].detach().cpu())
        
        preds_risk['ultrasound'].append(predictions_ultrasound[0])
        preds_subtype['ultrasound'].append(predictions_ultrasound[1])
        gts['risk'].append(labels[0].cpu().numpy())
        gts['subtype'].append(labels[1].cpu().numpy())
        
        predictions = {'ultrasound': predictions_ultrasound}

        for j, sample in enumerate(samples):
            save_snapshot_with_attention(opt, opt.save_path + '/attention_maps/', sample, opt.target, attention_ultrasound[-1][j][:, 0, 2:], None, None)
            save_predictions(j, sample, opt.target, opt.save_path + '/attention_maps/', predictions)
            save_contribution_scores(j, sample, opt.target, opt.save_path + '/attention_maps/', importance_ultrasound, 'ultrasound')

    for modality in preds_risk.keys():
        if len(preds_risk[modality]) > 0:
            preds_risk[modality] = torch.cat(preds_risk[modality], dim=0).cpu().numpy()
            preds_subtype[modality] = torch.cat(preds_subtype[modality], dim=0).cpu().numpy()

    for key in gts.keys():
        if gts[key] and len(gts[key]) > 0:
            gts[key] = np.concatenate(gts[key], axis=0)
            
    features['risk'] = torch.stack(features['risk'], dim=0).cpu().numpy()
    features['subtype'] = torch.stack(features['subtype'], dim=0).cpu().numpy()

    save_all_prediction(test_samples, preds_risk, preds_subtype, opt.save_path + 'ultrasound_predictions.xlsx')

    np.savez(
        opt.save_path + 'ultrasound_predictions.npz',
        preds_risk = preds_risk,
        preds_subtype = preds_subtype,
        features = features,
        gts=gts
    )