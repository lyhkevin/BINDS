import random
import json
from utils.dataloader import *
from utils.util import *
from utils.option import *

def stats(opt, samples):
    stats = get_count(opt.target)
    num_ultrasound, num_mammogram, num_mri, num_pathology = 0, 0, 0, 0
    for sample in samples:
        if sample['ultrasound'] is not None:
            num_ultrasound += 1
        if sample['mammogram'] is not None:
            num_mammogram += 1
        if sample['mri'] is not None:
            num_mri += 1
        if sample['pathology'] is not None:
            num_pathology += 1
        for i in range(len(opt.target)):
            stats[i][sample['labels'][i]] += 1
    for i in range(len(opt.target)):
        print(opt.target[i], stats[i])
    print('num_ultrasound:', num_ultrasound, 'num_mammogram:', num_mammogram,
          'num_mri:', num_mri, 'num_pathology:', num_pathology)

def split_by_ratio(items, train_ratio=0.7, val_ratio=0.1):
    n = len(items)
    train_len = int(n * train_ratio)
    val_len = int(n * val_ratio)
    train = items[:train_len]
    val = items[train_len:train_len + val_len]
    internal = items[train_len + val_len:]
    return train, val, internal

if __name__ == '__main__':

    seed_everything(42)
    opt = Options().get_opt()
    opt.modalities = ['mammogram', 'mri', 'ultrasound', 'pathology']
    opt.target = ['Risk', 'Subtype']
    opt.key_modalities = []
    opt.radiological = False
    opt.shuffle = False
    opt.split_data = True
    
    dataset = Dataset(mode=None, opt=opt)

    samples = dataset.samples
    public = [sample for sample in samples if 'public' in sample['cohort']]
    yunnan = [sample for sample in samples if 'yunnan' in sample['cohort']]
    wenzhou = [sample for sample in samples if 'wenzhou' in sample['cohort']]

    hangzhou = [sample for sample in samples if 'hangzhou' in sample['cohort']]
    shanghai = [sample for sample in samples if 'shanghai' in sample['cohort']]
    guizhou = [sample for sample in samples if 'guizhou' in sample['cohort']]

    xiangya = [sample for sample in samples if 'xiangya' in sample['cohort']]
    xinan = [sample for sample in samples if 'xinan' in sample['cohort']]
    fuyiyuan = [sample for sample in samples if 'fuyiyuan' in sample['cohort']]

    print('number of patients from public:', len(public))
    print('number of patients from yunnan:', len(yunnan))
    print('number of patients from wenzhou:', len(wenzhou))
    print('number of patients from xinan:', len(xinan))
    print('number of patients from guizhou:', len(guizhou))
    print('number of patients from fuyiyuan:', len(fuyiyuan))
    print('number of patients from hangzhou:', len(hangzhou))
    print('number of patients from shanghai:', len(shanghai))
    print('number of patients from xiangya:', len(xiangya))

    train_samples = []
    validation_samples = []
    internal_samples = []
    external_1_samples = []
    external_2_samples = []

    yunnan_path = [s for s in yunnan if s['pathology'] is not None]
    yunnan_no_path = [s for s in yunnan if s['pathology'] is None]

    random.shuffle(yunnan_path)
    random.shuffle(yunnan_no_path)

    yunnan_path_train, yunnan_path_val, yunnan_path_internal = split_by_ratio(yunnan_path, 0.7, 0.1)
    yunnan_no_path_train, yunnan_no_path_val, yunnan_no_path_internal = split_by_ratio(yunnan_no_path, 0.7, 0.1)

    yunnan_train = yunnan_path_train + yunnan_no_path_train
    yunnan_val = yunnan_path_val + yunnan_no_path_val
    yunnan_internal = yunnan_path_internal + yunnan_no_path_internal

    train_samples += public
    train_samples += yunnan_train
    train_samples += wenzhou

    validation_samples += yunnan_val
    internal_samples += yunnan_internal
    external_1_samples += hangzhou + shanghai + guizhou
    external_2_samples += xinan + fuyiyuan + xiangya

    print('---------------------------')
    print('train set stats:')
    stats(opt, train_samples)
    print('validation set stats:')
    stats(opt, validation_samples)
    print('internal stats:')
    stats(opt, internal_samples)
    print('external_1 stats:')
    stats(opt, external_1_samples)
    print('external_2 stats:')
    stats(opt, external_2_samples)

    train_samples = [(subject['cohort'], subject['id']) for subject in train_samples]
    validation_samples = [(subject['cohort'], subject['id']) for subject in validation_samples]
    internal_samples = [(subject['cohort'], subject['id']) for subject in internal_samples]
    external_1_samples = [(subject['cohort'], subject['id']) for subject in external_1_samples]
    external_2_samples = [(subject['cohort'], subject['id']) for subject in external_2_samples]

    data = {
        "train": train_samples,
        "validation": validation_samples,
        "internal": internal_samples,
        "external_1": external_1_samples,
        "external_2": external_2_samples
    }

    file_path = "../dataset/data.json"
    with open(file_path, 'w') as file:
        json.dump(data, file)
    print(f"Data splits saved to {file_path}")