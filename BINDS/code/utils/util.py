import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import shutil

num_classes = {}
num_classes['Risk'] = 2
num_classes['ER'] = 2
num_classes['PR'] = 2
num_classes['Ki67'] = 2
num_classes['HER2'] = 2
num_classes['Molecular_subtype'] = 4
num_classes['Luminal'] = 2
num_classes['Triple_negative'] = 2
num_classes['Subtype'] = 2
num_classes['Grade'] = 3
num_classes['Pre_treatment_T'] = 4
num_classes['Pre_treatment_N'] = 4
num_classes['Pre_treatment_M'] = 2
num_classes['Pre_treatment_stage'] = 8
num_classes['Post_treatment_T'] = 5
num_classes['Post_treatment_N'] = 4
num_classes['Post_treatment_M'] = 2
num_classes['Post_treatment_stage'] = 9
num_classes['pCR'] = 2

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_args_to_file(args, file_path):
    with open(file_path + 'arguments.txt', "w") as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
    shutil.copy('../dataset/data.json', file_path + "data.json")
        

def loss_function(criterions, predictions, labels):
    losses = []
    for criterion, prediction, label in zip(criterions, predictions, labels):
        mask = label != -1
        if len(label[mask]) == 0:
            continue
        prediction = F.softmax(prediction, dim=-1)
        prediction = prediction[mask]
        label = label[mask]
        loss = criterion(prediction, label)
        losses.append(loss)
    loss = sum(losses) / len(losses)
    return loss

def get_acc(predictions, labels):
    acc = []
    for prediction, label in zip(predictions, labels):
        _, prediction = torch.max(prediction, 1)
        mask = label != -1
        if len(label[mask]) == 0:
            acc.append(1)
            continue
        prediction = prediction[mask]
        label = label[mask]
        accuracy = torch.sum(prediction == label) * 1.0 / label.size()[0]
        acc.append(accuracy.item())
    return acc

def get_prob(pred, prob, gt, prediction, label):
    prediction = F.softmax(prediction, dim=1)
    prob.append(prediction.cpu().numpy())
    prediction = torch.argmax(prediction, dim=1)
    pred.append(prediction.cpu().numpy())
    gt.append(label.cpu().numpy())
    return pred, prob, gt

def get_class_names(target):
    class_names = {}
    class_names['Risk'] = ['Benign', 'Malignant']
    class_names['ER'] = ['ER-', 'ER+']
    class_names['PR'] = ['PR-', 'PR+']
    class_names['HER2'] = ['HER2-', 'HER2+']
    class_names['Ki67'] = ['Ki67-', 'Ki67+']
    class_names['Molecular_subtype'] = ['Luminal A', 'Luminal B', 'HER2+', 'TBNC']
    class_names['Luminal'] = ['Luminal', 'Non-luminal']
    class_names['Triple_negative'] = ['Non-TNBC', 'TNBC']
    class_names['Subtype'] = ['IDC', 'The rest']
    class_names['Histological_grade'] = ['Stage I', 'Stage II', 'Stage III']
    class_names['Pre_treatment_T'] = ['T1', 'T2', 'T3', 'T4']
    class_names['Pre_treatment_N'] = ['N0', 'N1', 'N2', 'N3']
    class_names['Pre_treatment_M'] = ['M0', 'M1']
    class_names['Pre_treatment_stage'] = ['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC', 'IV']
    class_names['Post_treatment_T'] = ['T0', 'T1', 'T2', 'T3', 'T4']
    class_names['Post_treatment_N'] = ['N0', 'N1', 'N2', 'N3']
    class_names['Post_treatment_M'] = ['M0', 'M1']
    class_names['Post_treatment_stage'] = ['0', 'IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC', 'IV']
    class_names['pCR'] = ['pCR-', 'pCR+']
    return class_names[target]

def get_num_class(attributes):
    num_class = []
    for attribute in attributes:
        num_class.append(num_classes[attribute])
    return num_class

def get_valid_values(attributes):
    valid_values = {}
    for attribute in attributes:
        valid_values[attribute] = [i for i in range(num_classes[attribute])]
    return valid_values

def get_oversample_rate(targets, rates):
    oversample_rates = {}
    for target, rate in zip(targets, rates):
        oversample_rates[target] = rate
    return oversample_rates

def get_count(attributes):
    counts = []
    for attribute in attributes:
        counts.append([0] * (num_classes[attribute] + 1))
    return counts