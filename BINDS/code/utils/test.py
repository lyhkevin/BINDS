import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
)
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

def bootstrap_auc(gt, pred, n_bootstraps=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    bootstrapped_aucs = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(pred), len(pred))
        if len(np.unique(gt[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(gt[indices], pred[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))
    return np.sort(bootstrapped_aucs)

def _bootstrap_ci(sample, low=2.5, high=97.5):
    return np.percentile(sample, [low, high])

def save_metrics_txt(metrics_ci, file_path):
    with open(file_path, "w") as f:
        for metric, (mean, low, high) in metrics_ci.items():
            f.write(f"{metric}: {mean:.3f} (95% CI {low:.3f}-{high:.3f})\n")

def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    accs, sens, specs, f1s = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_pred), len(y_pred))
        y_t, y_p = y_true[idx], y_pred[idx]
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

        accs.append((tp + tn) / (tp + tn + fp + fn))
        sens.append(tp / (tp + fn) if (tp + fn) else 0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0)
        f1s.append(f1_score(y_t, y_p))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc_true = (tp + tn) / (tp + tn + fp + fn)
    sens_true = tp / (tp + fn) if (tp + fn) else 0
    specs_true = tn / (tn + fp) if (tn + fp) else 0
    f1_true = f1_score(y_true, y_pred)

    return {
        "Accuracy": (acc_true, *_bootstrap_ci(accs)),
        "Sensitivity": (sens_true, *_bootstrap_ci(sens)),
        "Specificity": (specs_true, *_bootstrap_ci(specs)),
        "F1-score": (f1_true, *_bootstrap_ci(f1s)),
    }

def test_cancer_risk(preds, gts, save_path, title):
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    matplotlib.use("Agg")
    #plt.rcParams.update({"font.family": "Arial", "font.size": 14})
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(6, 6))

    y_score = preds[:, 1]
    y_true = gts

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ci_low, ci_high = np.percentile(
        bootstrap_auc(y_true, y_score),
        [2.5, 97.5]
    )

    y_pred_binary = (y_score >= 0.5).astype(int)
    metrics_ci = bootstrap_metrics(y_true, y_pred_binary)
    save_metrics_txt(metrics_ci, os.path.join(save_path, "cancer_risk_metrics.txt"))

    label = (
        f"Internal test cohort\n"
        f"(AUROC = {roc_auc:.3f}, 95% CI {ci_low:.3f}-{ci_high:.3f})"
    )

    plt.plot(fpr, tpr, lw=2, label=label)

    print("\nInternal test cohort:")
    for metric, (mean, low, high) in metrics_ci.items():
        print(f"{metric}: {mean:.3f} (95% CI {low:.3f}-{high:.3f})")
    print("-" * 30)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right", fontsize=11.5)
    plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect(0.9, adjustable="box")

    plt.savefig(os.path.join(save_path, f"{title}.png"), dpi=1000)
    plt.clf()

def test_cancer_subtype(preds, gts, save_path, title):
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    matplotlib.use("Agg")
    os.makedirs(save_path, exist_ok=True)

    mask = np.isin(gts, [0, 1])
    y_true = gts[mask]
    y_score = preds[mask, 1]

    plt.figure(figsize=(6, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ci_low, ci_high = np.percentile(
        bootstrap_auc(y_true, y_score),
        [2.5, 97.5]
    )

    y_pred_binary = (y_score >= 0.5).astype(int)
    metrics_ci = bootstrap_metrics(y_true, y_pred_binary)
    save_metrics_txt(metrics_ci, os.path.join(save_path, "cancer_subtype_metrics.txt"))

    label = (
        f"Internal test cohort\n"
        f"(AUROC = {roc_auc:.3f}, 95% CI {ci_low:.3f}-{ci_high:.3f})"
    )

    plt.plot(fpr, tpr, lw=2, label=label)

    print("\nInternal test cohort (Subtype binary):")
    print(f"Samples used: {len(y_true)}")
    for metric, (mean, low, high) in metrics_ci.items():
        print(f"{metric}: {mean:.3f} (95% CI {low:.3f}-{high:.3f})")
    print("-" * 30)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right", fontsize=11.5)
    plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect(0.9, adjustable="box")

    plt.savefig(os.path.join(save_path, f"{title}.png"), dpi=1000)
    plt.clf()
    
def tsne_plot(features, gt, title, file_name, s=10, alpha=0.3):
    gt_np = np.array(gt)
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)

    plt.figure(figsize=(6, 4))
    # plt.rcParams['font.family'] = 'Arial'

    benign_mask = (gt_np == 0)
    malignant_mask = (gt_np == 1)

    plt.scatter(tsne[benign_mask, 0], tsne[benign_mask, 1],
                c='green', label='Benign', alpha=alpha, s=s)
    plt.scatter(tsne[malignant_mask, 0], tsne[malignant_mask, 1],
                c='red', label='Malignant', alpha=alpha, s=s)

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='Benign',
               markerfacecolor='green', markeredgecolor='none', markersize=7),
        Line2D([0], [0], marker='o', color='none', label='Malignant',
               markerfacecolor='red', markeredgecolor='none', markersize=7)
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
    plt.title(title, fontsize=19)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_name, dpi=1000)
    plt.close()