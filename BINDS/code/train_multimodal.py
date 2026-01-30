from models.mri_encoder import *
from models.ultrasound_encoder import *
from models.mammogram_encoder import *
from models.multimodal_predictor import *
from utils.save_img import *
import logging
from utils.dataloader import *
from utils.validation import *
from utils.util import *
from utils.focal_loss import *
from utils.option import *
from tqdm import tqdm

if __name__ == '__main__':
    seed_everything(42)
    opt = Options().get_opt()
    opt.multimodal = True
    opt.accumulation_steps = 4
    opt.weight_save_interval = 1
    opt.epoch = 10
    opt.lr = 1e-5
    opt.min_lr = 1e-7
    opt.warmup_epochs = 1
    opt.batch_size = 5
    opt.num_workers = 4
    opt.modalities = ['mammogram', 'ultrasound', 'mri']
    opt.min_num_modalities = 2
    opt.random_mask = True
    opt.mask_ratio = 0.5
    opt.num_heads = 6
    opt.depth_fusion = 6
    opt.weight_save_path = '../weight/multi_modal/'
    opt.img_save_path = opt.weight_save_path + 'snapshot/'
    opt.target = ['Risk', 'Subtype']
    opt.oversample = True
    opt.augmentation = True
    opt.oversample_targets = ['Risk', 'Subtype']
    opt.oversample_rates = [[10, 0], [0, 1]]
    opt.oversample_modality = {'us+mm': 4, 'us+mri': 0, 'mm+mri': 2, 'us+mm+mri': 2}
    opt.loss_weight = [[10.0, 1.0], [1.0, 20.0]]
    opt.save_snapshot = True
    opt.snapshot_interval = 1000
    opt.acc_threshold = [1, 1]

    opt.validation_interval = 1 
    best_avg_auc = 0.0
    best_epoch = 0

    os.makedirs(opt.weight_save_path, exist_ok=True)
    os.makedirs(opt.img_save_path, exist_ok=True)
    save_args_to_file(opt, opt.weight_save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader = get_dataloader(batch_size=opt.batch_size, shuffle=opt.shuffle, pin_memory=False,
                                                 num_workers=opt.num_workers, mode='train', opt=opt)
    validation_dataset, validation_loader = get_dataloader(batch_size=opt.batch_size, shuffle=opt.shuffle, pin_memory=False,
                                                           num_workers=opt.num_workers, mode='validation', opt=opt)
    num_classes = train_dataset.num_classes

    criterions = []
    for i, weight in enumerate(opt.loss_weight):
        if weight == None:
            criterion = FocalLoss(gamma=0.7)
        else:
            criterion = FocalLoss(gamma=0.7, weights=torch.FloatTensor(weight).to(device))
        criterions.append(criterion)

    print('start training')
    logging.basicConfig(filename=opt.weight_save_path + 'train.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO,
                        filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    encoder_ultrasound = ultrasound_encoder(embed_dim=opt.embed_dim, backbone=opt.backbone_ultrasound,
                                            num_classes=num_classes, depth_encoder=opt.depth_encoder_ultrasound,
                                            depth_fusion=opt.depth_fusion_ultrasound, multi_modal=True).to(device)
    encoder_mammogram = mammogram_encoder(embed_dim=opt.embed_dim, backbone=opt.backbone_mammogram,
                                          depth_encoder=opt.depth_encoder_mammogram, num_classes=num_classes,
                                          depth_fusion=opt.depth_fusion_mammogram, multi_modal=True).to(device)
    encoder_mri = mri_encoder(modalities=opt.mri_modalities, embed_dim=opt.embed_dim, backbone=opt.backbone_mri, pretrain_path=opt.ResNet_3D_path,
                              num_classes=num_classes, depth_encoder=opt.depth_encoder_mri, depth_fusion=opt.depth_fusion_mri,
                              multi_modal=True).to(device)

    encoder_ultrasound.load_state_dict(torch.load(opt.weight_save_path_ultrasound), strict=False)
    encoder_mammogram.load_state_dict(torch.load(opt.weight_save_path_mammogram), strict=False)
    encoder_mri.load_state_dict(torch.load(opt.weight_save_path_mri), strict=False)
    
    predictor = multimodal_predictor(device=device, num_heads=opt.num_heads, num_classes=num_classes,
                                     embed_dim=opt.embed_dim, depth=opt.depth_fusion, dropout=0.4).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=opt.lr)
    lr_scheduler = cosine_scheduler(opt.lr, opt.min_lr, opt.epoch, len(train_loader), warmup_epochs=opt.warmup_epochs)

    encoder_mri.eval()
    encoder_ultrasound.eval()
    encoder_mammogram.eval()
    predictor.train()

    for epoch in range(0, opt.epoch):
        epoch_acc = []
        optimizer.zero_grad()
        for i, samples in enumerate(tqdm(train_loader)):
            it = len(train_loader) * epoch + i
            for j, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[it]
            if opt.save_snapshot == True and i % opt.snapshot_interval == 0:
                save_snapshot(opt=opt, save_path=opt.img_save_path, sample=samples[0], target=opt.target)

            imgs_mri, imgs_ultrasound, imgs_mammogram = [], [], []
            has_mri, has_ultrasound, has_mammogram = [], [], []
            labels = []
            for sample in samples:
                has_mri.append(sample['has_mri'])
                has_ultrasound.append(sample['has_ultrasound'])
                has_mammogram.append(sample['has_mammogram'])
                imgs_mri.append(sample['mri'])
                imgs_ultrasound.append(sample['ultrasound'])
                imgs_mammogram.append(sample['mammogram'])
                labels.append(sample['labels'])

            imgs_mri = torch.stack(imgs_mri, dim=0).to(device)
            imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0).to(device)
            imgs_mammogram = torch.stack(imgs_mammogram, dim=0).to(device)
            labels = torch.stack(labels, dim=1).to(device)
            
            with torch.no_grad():
                f_ultrasound, _, _ = encoder_ultrasound(imgs_ultrasound)
                f_mammogram, _, _ = encoder_mammogram(imgs_mammogram)
                f_mri, _, _ = encoder_mri(imgs_mri)
            
            _, predictions = predictor(has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri)

            loss = loss_function(criterions, predictions, labels)
            loss = loss / opt.accumulation_steps
            loss.backward()
            
            if (i + 1) % opt.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            acc = get_acc(predictions, labels)
            acc_str = ', '.join('%s: %.2f' % (label, accuracy) for label, accuracy in zip(opt.target, acc))
            epoch_acc.append(acc)
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [acc: %s] [lr: %f]" % (
                epoch, opt.epoch, i, len(train_loader), loss.item() * opt.accumulation_steps, acc_str, get_lr(optimizer)))
            if (i + 1) % opt.log_interval == 0:
                logging.info("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [acc: %s] [lr: %f]" % (
                    epoch, opt.epoch, i, len(train_loader), loss.item() * opt.accumulation_steps, acc_str, get_lr(optimizer)))

        if (epoch + 1) % opt.validation_interval == 0:
            print(f'start validation at epoch {epoch + 1}')
            predictor.eval()
            pred = [[] for _ in range(len(opt.target))]
            gt = [[] for _ in range(len(opt.target))]
            for _, samples in enumerate(tqdm(validation_loader)):
                has_mri, has_ultrasound, has_mammogram = [], [], []
                imgs_mri, imgs_ultrasound, imgs_mammogram = [], [], []
                labels = []
                for sample in samples:
                    has_mri.append(sample['has_mri']); has_ultrasound.append(sample['has_ultrasound']); has_mammogram.append(sample['has_mammogram'])
                    imgs_mri.append(sample['mri']); imgs_ultrasound.append(sample['ultrasound']); imgs_mammogram.append(sample['mammogram'])
                    labels.append(sample['labels'])
                
                imgs_mri = torch.stack(imgs_mri, dim=0).to(device)
                imgs_ultrasound = torch.stack(imgs_ultrasound, dim=0).to(device)
                imgs_mammogram = torch.stack(imgs_mammogram, dim=0).to(device)
                labels = torch.stack(labels, dim=1).to(device)
                
                with torch.no_grad():
                    f_ultrasound, _, _ = encoder_ultrasound(imgs_ultrasound)
                    f_mammogram, _, _ = encoder_mammogram(imgs_mammogram)
                    f_mri, _, _ = encoder_mri(imgs_mri)
                    _, predictions = predictor(has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri)
                
                for j in range(len(opt.target)):
                    pred[j].append(predictions[j].cpu())
                    gt[j].append(labels[j].cpu().numpy())

            current_aucs = []
            for j, target in enumerate(opt.target):
                auc_val = validation(target=target, pred=pred[j], gt=gt[j], num_classes=num_classes[j], opt=opt, epoch=epoch+1)
                current_aucs.append(auc_val)
            
            avg_auc = sum(current_aucs) / len(current_aucs)
            print(f'Epoch {epoch + 1} Average AUC: {avg_auc:.4f}')

            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_epoch = epoch + 1
                torch.save(predictor.state_dict(), opt.weight_save_path + 'best_predictor.pth')
                print(f'New best model saved with Average AUC: {best_avg_auc:.4f}')
                logging.info(f'---> New best model saved at Epoch {best_epoch} with Average AUC: {best_avg_auc:.4f}')
            
            predictor.train()
        
        if (epoch + 1) % opt.weight_save_interval == 0:
            torch.save(predictor.state_dict(), opt.weight_save_path + 'predictor_' + str(epoch + 1) + '.pth')

        epoch_acc = torch.tensor(epoch_acc)
        epoch_acc = torch.mean(epoch_acc, dim=0)
        print('Task-wise average accuracies:')
        for i, (task, acc_val, threshold) in enumerate(zip(opt.target, epoch_acc, opt.acc_threshold)):
            print(f'{task}: {acc_val:.4f} (threshold: {threshold})')
        if all(acc_val > threshold for acc_val, threshold in zip(epoch_acc, opt.acc_threshold)):
            print('All task accuracies exceed thresholds, training finished.')
            break
        
    torch.save(predictor.state_dict(), opt.weight_save_path + 'predictor.pth')
    
    final_msg = f'Training Finished. Best model was saved from Epoch {best_epoch} with Average AUC {best_avg_auc:.4f}'
    print(final_msg)
    logging.info(final_msg)