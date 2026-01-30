import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.test import *
import numpy as np

root = '/public/home/liyh2022/projects/BINDS/evaluation/two_stage/'
data = np.load(os.path.join(root, 'two_stage_predictions.npz'), allow_pickle=True)

preds_risk = data['preds_risk'].item()['final']
preds_subtype = data['preds_subtype'].item()['final']

features_risk = data['features'].item()['risk']
features_subtype = data['features'].item()['subtype']

gts_risk = data['gts'].item()['risk']
gts_subtype = data['gts'].item()['subtype']

save_root = os.path.join(root, 'test')
os.makedirs(save_root, exist_ok=True)

risk_save_path = os.path.join(save_root, 'risk')
test_cancer_risk(
    preds=preds_risk,
    gts=gts_risk,
    save_path=risk_save_path,
    title='Two_stage_Risk'
)

tsne_plot(
    features=features_risk,
    gt=gts_risk,
    title='Two_stage Risk t-SNE',
    file_name=os.path.join(risk_save_path, 'tsne_risk.png'),
    s=10,
    alpha=0.3
)

subtype_save_path = os.path.join(save_root, 'subtype')
test_cancer_subtype(
    preds=preds_subtype,
    gts=gts_subtype,
    save_path=subtype_save_path,
    title='Two_stage_Subtype'
)

tsne_plot(
    features=features_subtype,
    gt=gts_subtype,
    title='Two_stage Subtype t-SNE',
    file_name=os.path.join(subtype_save_path, 'tsne_subtype.png'),
    s=10,
    alpha=0.3
)

print('Evaluation finished.')