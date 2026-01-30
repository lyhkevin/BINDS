from models.pathology_encoder import *
from utils.save_img import *
import logging
from utils.dataloader import *
from utils.validation import *
from utils.util import *
from utils.focal_loss import *
from utils.option import *
from tqdm import tqdm

def compute_orthogonal_loss(features):
    feats = []
    for f in features:
        B, D = f.size()
        f = nn.functional.normalize(f, p=2, dim=1)
        sim_matrix = torch.matmul(f, f.T)
        I = torch.eye(B, device=f.device)
        feats.append(((sim_matrix - I) ** 2).mean())
    return torch.stack(feats).mean()

if __name__ == '__main__':
    seed_everything(42)
    opt = Options().get_opt()
    opt.accumulation_steps = 4
    opt.lr = 1e-4
    opt.min_lr = 1e-7
    opt.warmup_epochs = 10
    opt.batch_size = 10
    opt.num_workers = 4
    opt.modalities = ['pathology']
    opt.key_modalities = ['pathology']
    opt.DINOv2_path = '../weight/pretrain/dinov2.pth'
    opt.weight_save_path = '../weight/pathology_final/'
    opt.img_save_path = opt.weight_save_path + 'snapshot/'
    opt.radiological = False
    opt.target = ['Subtype']
    opt.oversample = True
    opt.augmentation = True
    opt.oversample_targets = ['Subtype']
    opt.oversample_rates = [[0, 50]]
    opt.loss_weight = [[1.0, 20.0]]
    opt.backbone_pathology = 'dinov2'
    opt.pathology_scale = ['small', 'medium', 'large']  
    opt.save_snapshot = True
    opt.snapshot_interval = 1000
    opt.acc_threshold = [0.99]

    opt.validation_interval = 5
    best_avg_auc = 0.0
    best_epoch = 0

    os.makedirs(opt.weight_save_path, exist_ok=True)
    os.makedirs(opt.img_save_path, exist_ok=True)
    save_args_to_file(opt, opt.weight_save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader = get_dataloader(batch_size=opt.batch_size, shuffle=opt.shuffle, pin_memory=False,
                                                 num_workers=opt.num_workers, mode='train', opt=opt)
    internal_dataset, internal_loader = get_dataloader(batch_size=opt.batch_size, shuffle=opt.shuffle, pin_memory=False,
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
    
    encoder = Pathology_encoder(scale=opt.pathology_scale, embed_dim=opt.embed_dim_pathology, num_classes=num_classes,
                                device=device, backbone=opt.backbone_pathology,
                                depth_encoder=opt.depth_encoder_pathology, depth_fusion=opt.depth_fusion_pathology,
                                pretrain_path=opt.DINOv2_path).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    lr_scheduler = cosine_scheduler(opt.lr, opt.min_lr, opt.epoch, len(train_loader), warmup_epochs=opt.warmup_epochs)
    encoder.train()

    for epoch in range(0, opt.epoch):
        epoch_acc = []
        optimizer.zero_grad()
        
        for i, samples in enumerate(tqdm(train_loader)):
            it = len(train_loader) * epoch + i
            for j, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[it]
            if opt.save_snapshot == True and i % opt.snapshot_interval == 0:
                save_snapshot(opt=opt, save_path=opt.img_save_path, sample=samples[0], target=opt.target)

            imgs = []
            labels = []
            for sample in samples:
                imgs.append(sample['pathology'])
                labels.append(sample['labels'])
            imgs = torch.stack(imgs, dim=0).to(device)
            labels = torch.stack(labels, dim=1).to(device)

            small_imgs, medium_imgs, large_imgs = torch.chunk(imgs, chunks=3, dim=1)
            small_imgs, medium_imgs, large_imgs = small_imgs.to(device), medium_imgs.to(device), large_imgs.to(device)
            predictions, features = encoder(small_imgs, medium_imgs, large_imgs)
            orthogonal_loss = compute_orthogonal_loss(features)
            loss = loss_function(criterions, predictions, labels) + 0.05 * orthogonal_loss
            loss.backward()

            if (i + 1) % opt.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            acc = get_acc(predictions, labels)
            acc_str = ', '.join('%s: %.2f' % (label, accuracy) for label, accuracy in zip(opt.target, acc))
            epoch_acc.append(acc)
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [orthogonal: %f] [acc: %s] [lr: %f]" % (
                epoch, opt.epoch, i, len(train_loader), loss.item(), orthogonal_loss.item(), acc_str, get_lr(optimizer)))
            if (i + 1) % opt.log_interval == 0:
                logging.info("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [orthogonal: %f] [acc: %s] [lr: %f]" % (
                    epoch, opt.epoch, i, len(train_loader), loss.item(), orthogonal_loss.item(), acc_str, get_lr(optimizer)))

        if (epoch + 1) % opt.validation_interval == 0:
            print(f'start validation at epoch {epoch + 1}')
            encoder.eval()
            pred = [[] for _ in range(len(opt.target))]
            gt = [[] for _ in range(len(opt.target))]
            for _, samples in enumerate(tqdm(internal_loader)):
                imgs = []
                labels = []
                for sample in samples:
                    imgs.append(sample['pathology'])
                    labels.append(sample['labels'])
                imgs = torch.stack(imgs, dim=0).to(device)
                labels = torch.stack(labels, dim=1).to(device)
                
                small_imgs, medium_imgs, large_imgs = torch.chunk(imgs, chunks=3, dim=1)
                with torch.no_grad():
                    predictions, _ = encoder(small_imgs.to(device), medium_imgs.to(device), large_imgs.to(device))
                
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
                torch.save(encoder.state_dict(), opt.weight_save_path + 'best_encoder_pathology.pth')
                print(f'New best model saved with Average AUC: {best_avg_auc:.4f}')
                logging.info(f'---> New best model saved at Epoch {best_epoch} with Average AUC: {best_avg_auc:.4f}')
            
            encoder.train()

        if (epoch + 1) % opt.weight_save_interval == 0:
            torch.save(encoder.state_dict(), opt.weight_save_path + 'encoder_pathology_' + str(epoch + 1) + '.pth')
        
        epoch_acc = torch.tensor(epoch_acc)
        epoch_acc = torch.mean(epoch_acc, dim=0)
        print('Task-wise average accuracies:')
        for i, (task, acc_val, threshold) in enumerate(zip(opt.target, epoch_acc, opt.acc_threshold)):
            print(f'{task}: {acc_val:.4f} (threshold: {threshold})')
        if all(acc_val > threshold for acc_val, threshold in zip(epoch_acc, opt.acc_threshold)):
            print('All task accuracies exceed thresholds, training finished.')
            break

    torch.save(encoder.state_dict(), opt.weight_save_path + 'encoder_pathology.pth')
    
    final_msg = f'Training Finished. Best model was saved from Epoch {best_epoch} with Average AUC {best_avg_auc:.4f}'
    print(final_msg)
    logging.info(final_msg)