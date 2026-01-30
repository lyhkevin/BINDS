import seaborn as sns
import matplotlib.pyplot as plt
from utils.util import *
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import numpy as np

def mean_ci(x):
    x = np.array(x)
    x = x[~np.isnan(x)]
    ci_lower = np.percentile(x, 2.5)
    ci_upper = np.percentile(x, 97.5)
    return (ci_lower, ci_upper)

def plot_confusion_matrix(cm, path, names):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm / np.sum(cm,axis=1), annot=True, fmt='.1%', cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    plt.savefig(path + '/confusion_matrix.png')
    plt.clf()

def bootstrap_metrics(gt, pred_label, prob, num_classes=2, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    gt = np.array(gt)
    pred_label = np.array(pred_label)
    prob = np.array(prob)
    metrics = {}
    ci_dict = {}

    metrics["Accuracy"] = accuracy_score(gt, pred_label)
    metrics["Sensitivity"] = recall_score(gt, pred_label, pos_label=1)
    metrics["F1-score"] = f1_score(gt, pred_label, pos_label=1)

    if num_classes == 2:
        cm = confusion_matrix(gt, pred_label, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = np.nan
        fpr, tpr, _ = roc_curve(gt, prob[:, 1])
        auc_score = auc(fpr, tpr)
        metrics["Specificity"] = specificity
        metrics["AUC"] = auc_score
    else:
        metrics["Specificity"] = np.nan
        metrics["AUC"] = np.nan

    accs, recalls, specs, f1s, aucs = [], [], [], [], []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(gt), len(gt))
        gti = gt[indices]
        predi = pred_label[indices]
        probi = prob[indices]

        if len(np.unique(gti)) < 2:
            continue

        accs.append(accuracy_score(gti, predi))
        recalls.append(recall_score(gti, predi, pos_label=1))
        f1s.append(f1_score(gti, predi, pos_label=1))

        if num_classes == 2:
            cm = confusion_matrix(gti, predi, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                specs.append(specificity)
            else:
                specs.append(np.nan)

            fpr, tpr, _ = roc_curve(gti, probi[:, 1])
            aucs.append(auc(fpr, tpr))
        else:
            specs.append(np.nan)
            aucs.append(np.nan)

    ci_dict["Accuracy"] = mean_ci(accs)
    ci_dict["Sensitivity"] = mean_ci(recalls)
    ci_dict["Specificity"] = mean_ci(specs)
    ci_dict["F1-score"] = mean_ci(f1s)
    ci_dict["AUC"] = mean_ci(aucs)
    return metrics, ci_dict

def validation(target, pred, gt, num_classes, opt, epoch):
    names = get_class_names(target)
    save_path = os.path.join(opt.weight_save_path, target, f'epoch_{epoch}')
    os.makedirs(save_path, exist_ok=True)
    prediction = torch.cat(pred, dim=0)
    prediction = F.softmax(prediction, dim=-1)
    probability = prediction.numpy()
    gt = np.concatenate(gt, axis=0)
    mask = gt != -1
    gt = gt[mask]
    probability = probability[mask]
    prediction_label = np.argmax(probability, axis=-1)
    if len(set(gt)) == len(names):
        report = classification_report(gt, prediction_label, target_names=names)
        with open(os.path.join(save_path, 'classification_report.txt'), "w") as f:
            f.write(report)
        cm = confusion_matrix(gt, prediction_label)
        plot_confusion_matrix(cm, save_path, names)
        metrics, cis = bootstrap_metrics(gt, prediction_label, probability, num_classes)
        with open(os.path.join(save_path, 'metrics_summary.txt'), "w") as f:
            for key in metrics:
                mean = metrics[key]
                low, high = cis[key]
                f.write(f"{key}: {mean:.3f} (95% CI {low:.3f} - {high:.3f})\n")
    return metrics['AUC']