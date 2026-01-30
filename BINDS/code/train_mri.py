import torch
import os
import logging
from tqdm import tqdm
import torch.utils.data as data
from models.mri_encoder import *
from models.pathology_encoder import *
from models.alignment_loss import *
from utils.save_img import *
from utils.dataloader import *
from utils.validation import *
from utils.util import *
from utils.focal_loss import *
from utils.option import *

torch.multiprocessing.set_sharing_strategy('file_system')

def collate_fn(batch):
    return [*batch]

if __name__ == '__main__':
    seed_everything(42)
    opt = Options().get_opt()

    opt.alignment = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.lr = 1e-4
    opt.min_lr = 1e-7
    opt.warmup_epochs = 5
    opt.batch_size = 5
    opt.mri_view = 'Axial'
    opt.mri_modalities = ['ADC', 'P0', 'P2', 'T2']
    opt.target = ['Risk', 'Subtype']
    opt.backbone_mri = 'ResNet_3D'
    opt.oversample = True
    opt.augmentation = True
    opt.oversample_targets = ['Risk', 'Subtype']
    opt.save_snapshot = True
    opt.snapshot_interval = 1000
    opt.acc_threshold = [0.98, 0.98]
    
    opt.key_modalities = ['mri']
    opt.weight_save_path = '../weight/mri/'
    opt.img_save_path = '../snapshot/mri/'
    
    opt.accumulation_steps = 4
    opt.loss_weight = [[20.0, 1.0], [1.0, 20.0]]
    opt.oversample_rates = [[2, 0], [0, 1]]

    opt.validation_interval = 5
    best_avg_auc = 0.0
    best_epoch = 0

    if opt.alignment:
        opt.modalities = ['mri', 'pathology']
        opt.batch_size_alignment = 5
        opt.epoch_alignment = 10
    else:
        opt.modalities = ['mri']
        opt.img_save_path = opt.weight_save_path + 'snapshot/'

    os.makedirs(opt.weight_save_path, exist_ok=True)
    os.makedirs(opt.img_save_path, exist_ok=True)
    save_args_to_file(opt, opt.weight_save_path)

    current_batch_size = opt.batch_size_alignment if opt.alignment else opt.batch_size

    train_dataset, train_loader = get_dataloader(
        batch_size=current_batch_size, shuffle=opt.shuffle,
        pin_memory=False, num_workers=opt.num_workers, mode='train', opt=opt
    )
    validation_dataset, validation_loader = get_dataloader(
        batch_size=opt.batch_size, shuffle=opt.shuffle,
        pin_memory=False, num_workers=opt.num_workers,
        mode='validation', opt=opt
    )
    num_classes = train_dataset.num_classes

    criterions = []
    for i, weight in enumerate(opt.loss_weight):
        if weight is None:
            criterion = FocalLoss(gamma=0.7)
        else:
            criterion = FocalLoss(gamma=0.7, weights=torch.FloatTensor(weight).to(device))
        criterions.append(criterion)

    print('start training')
    logging.basicConfig(
        filename=opt.weight_save_path + 'train.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO,
        filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    encoder = mri_encoder(
        modalities=opt.mri_modalities, embed_dim=opt.embed_dim, backbone=opt.backbone_mri,
        num_classes=num_classes, depth_encoder=opt.depth_encoder_mri, pretrain_path=opt.ResNet_3D_path,
        depth_fusion=opt.depth_fusion_mri,
        embed_dim_pathology=opt.embed_dim_pathology,
        alignment=opt.alignment, multi_modal=False
    ).to(device)

    if opt.alignment:
        if hasattr(encoder, 'encoders'):
            for e in encoder.encoders:
                for param in encoder.encoders[e].parameters():
                    param.requires_grad = False
        elif hasattr(encoder, 'encoder'):
             for param in encoder.encoder.parameters():
                param.requires_grad = False
        else:
            for param in encoder.parameters():
                param.requires_grad = False

        encoder_pathology = Pathology_encoder(
            scale=opt.pathology_scale, embed_dim=opt.embed_dim_pathology,
            num_classes=[2], device=device, backbone=opt.backbone_pathology,
            depth_encoder=opt.depth_encoder_pathology, depth_fusion=opt.depth_fusion_pathology,
            alignment=True
        ).to(device)
        encoder_pathology.eval()
        encoder_pathology.load_state_dict(torch.load(opt.weight_save_path_pathology), strict=False)

        lr_scheduler = cosine_scheduler(opt.lr, opt.min_lr, opt.epoch_alignment, len(train_loader), warmup_epochs=opt.warmup_epochs)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=opt.lr)
    else:
        lr_scheduler = cosine_scheduler(
            opt.lr, opt.min_lr, opt.epoch, len(train_loader),
            warmup_epochs=opt.warmup_epochs
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    encoder.train()
    scheduler_step = 0

    for epoch in range(0, opt.epoch):
        epoch_acc = []
        if hasattr(train_dataset, 'epoch'):
            train_dataset.epoch = epoch

        if opt.alignment and epoch == opt.epoch_alignment:
            print("Alignment phase finished. Unfreezing parameters and resetting scheduler.")
            if hasattr(encoder, 'encoders'):
                for e in encoder.encoders:
                    for param in encoder.encoders[e].parameters():
                        param.requires_grad = True
            elif hasattr(encoder, 'encoder'):
                for param in encoder.encoder.parameters():
                    param.requires_grad = True
            else:
                for param in encoder.parameters():
                    param.requires_grad = True

            train_loader = data.DataLoader(
                dataset=train_dataset, batch_size=opt.batch_size,
                num_workers=opt.num_workers, shuffle=opt.shuffle,
                pin_memory=False, collate_fn=collate_fn
            )
            
            optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

            lr_scheduler = cosine_scheduler(
                opt.lr, opt.min_lr, opt.epoch - opt.epoch_alignment, len(train_loader),
                warmup_epochs=0 
            )
            
            scheduler_step = 0

        optimizer.zero_grad()

        for i, samples in enumerate(tqdm(train_loader)):
            if lr_scheduler is not None and scheduler_step < len(lr_scheduler):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_scheduler[scheduler_step]

            if opt.save_snapshot is True and i % opt.snapshot_interval == 0:
                save_snapshot(opt=opt, save_path=opt.img_save_path, sample=samples[0], target=opt.target)

            imgs = []
            labels = []

            if opt.alignment:
                imgs_pathology = []
                has_img = []
                has_pathology = []
                for sample in samples:
                    imgs.append(sample['mri'])
                    has_img.append(sample['has_mri'])
                    if sample['has_pathology']:
                        imgs_pathology.append(sample['pathology'])
                    has_pathology.append(sample['has_pathology'])
                    labels.append(sample['labels'])
            else:
                for sample in samples:
                    imgs.append(sample['mri'])
                    labels.append(sample['labels'])

            imgs = torch.stack(imgs, dim=0).to(device)
            labels = torch.stack(labels, dim=1).to(device)

            if opt.alignment:
                predictions, features_radiology = encoder(imgs)
                cls_loss = loss_function(criterions, predictions, labels)
                alignment_loss = None
                if any(has_pathology):
                    imgs_pathology = torch.stack(imgs_pathology, dim=0).to(device)
                    small_imgs, medium_imgs, large_imgs = torch.chunk(imgs_pathology, chunks=3, dim=1)
                    small_imgs, medium_imgs, large_imgs = small_imgs.to(device), medium_imgs.to(device), large_imgs.to(device)
                    with torch.no_grad():
                        _, features_pathology = encoder_pathology(small_imgs, medium_imgs, large_imgs)

                    alignment_loss = feature_alignment(features_radiology, features_pathology, has_img, has_pathology)

                    if alignment_loss is not None:
                        if epoch < opt.epoch_alignment:
                            cls_loss = loss_function(
                                [criterions[1]],    
                                [predictions[1]],  
                                [labels[1]]        
                            )
                    loss = cls_loss + 0.1 * alignment_loss
                else:
                    loss = cls_loss
            else:
                predictions = encoder(imgs)
                loss = loss_function(criterions, predictions, labels)

            loss = loss / opt.accumulation_steps
            loss.backward()

            if (i + 1) % opt.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            acc = get_acc(predictions, labels)
            acc_str = ', '.join('%s: %.2f' % (label, accuracy) for label, accuracy in zip(opt.target, acc))
            epoch_acc.append(acc)

            current_lr = get_lr(optimizer)
            if opt.alignment and any(has_pathology):
                if alignment_loss is not None:
                    msg = ("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [cls_loss: %f] [alignment_loss: %f] [acc: %s] [lr: %f]"
                           % (epoch, opt.epoch, i, len(train_loader),
                              (loss.item() * opt.accumulation_steps),
                              cls_loss.item(),
                              alignment_loss.item() if hasattr(alignment_loss, "item") else float(alignment_loss),
                              acc_str, current_lr))
                else:
                    msg = ("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [cls_loss: %f] [acc: %s] [lr: %f]"
                           % (epoch, opt.epoch, i, len(train_loader),
                              (loss.item() * opt.accumulation_steps),
                              cls_loss.item(),
                              acc_str, current_lr))
            else:
                msg = ("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [acc: %s] [lr: %f]"
                       % (epoch, opt.epoch, i, len(train_loader),
                          (loss.item() * opt.accumulation_steps),
                          acc_str, current_lr))

            print(msg)
            if (i + 1) % opt.log_interval == 0:
                logging.info(msg)

            if lr_scheduler is not None:
                scheduler_step += 1
                
        if (epoch + 1) % opt.validation_interval == 0:
            print(f'start validation at epoch {epoch + 1}')
            encoder.eval()
            pred = [[] for _ in range(len(opt.target))]
            gt = [[] for _ in range(len(opt.target))]
            for _, samples in enumerate(tqdm(validation_loader)):
                imgs = []
                labels = []
                for sample in samples:
                    imgs.append(sample['mri'])
                    labels.append(sample['labels'])
                imgs = torch.stack(imgs, dim=0).to(device)
                labels = torch.stack(labels, dim=1).to(device)
                with torch.no_grad():
                    if opt.alignment:
                        predictions, _ = encoder(imgs)
                    else:
                        predictions = encoder(imgs)
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
                torch.save(encoder.state_dict(), opt.weight_save_path + 'best_encoder_mri.pth')
                print(f'New best model saved with Average AUC: {best_avg_auc:.4f}')
                logging.info(f'---> New best model saved at Epoch {best_epoch} with Average AUC: {best_avg_auc:.4f}')
            
            encoder.train()

        if (epoch + 1) % opt.weight_save_interval == 0:
            torch.save(encoder.state_dict(), opt.weight_save_path + 'encoder_mri_' + str(epoch + 1) + '.pth')

        epoch_acc = torch.tensor(epoch_acc)
        epoch_acc = torch.mean(epoch_acc, dim=0)
        print('Task-wise average accuracies:')
        for i, (task, acc_val, threshold) in enumerate(zip(opt.target, epoch_acc, opt.acc_threshold)):
            print(f'{task}: {acc_val:.4f} (threshold: {threshold})')
        if all(acc_val > threshold for acc_val, threshold in zip(epoch_acc, opt.acc_threshold)):
            print('All task accuracies exceed thresholds, training finished.')
            break

    torch.save(encoder.state_dict(), opt.weight_save_path + 'encoder_mri.pth')
    
    final_msg = f'Training Finished. Best model was saved from Epoch {best_epoch} with Average AUC {best_avg_auc:.4f}'
    print(final_msg)
    logging.info(final_msg)