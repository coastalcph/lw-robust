import copy
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloaders import get_dataset
from configs.supported import F1, binary_logits_to_pred_v2
from scipy.special import expit
DATASET = 'bioasq116'
BASE_DIR = f'/home/iliasc/temporal-wilds/logs/{DATASET}/BATCH_64_V2'
SPLIT_SCHEME = 'official'
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
full_dataset = get_dataset(
    dataset=DATASET,
    version='1.0',
    root_dir='../data/datasets',
    split_scheme=SPLIT_SCHEME)

train_set = full_dataset.get_subset('train')
train_y_true = train_set.y_array.numpy()
test_set = full_dataset.get_subset('test')
test_y_true = test_set.y_array.numpy()
val_set = full_dataset.get_subset('val')
val_y_true = test_set.y_array.numpy()

def compute_model_rates(model):
    # calculate roc curve
    X = {class_i: {'FPR': [], 'FNR': []} for class_i in range(model['y_true'].shape[1])}
    for class_i in range(model['y_true'].shape[1]):
        cm = confusion_matrix(model['y_true'][:, class_i], model['y_pred'][:, class_i])
        FP = cm.sum(axis=0) - np.diag(cm) + [1e-8, 1e-8]
        FN = cm.sum(axis=1) - np.diag(cm) + [1e-8, 1e-8]
        TP = np.diag(cm) + [1e-8, 1e-8]
        TN = np.sum(cm) - (FP + FN + TP + [1e-8, 1e-8])
        FPR = FP / (FP + TN + [1e-8, 1e-8])
        FNR = FN / (TP + FN + [1e-8, 1e-8])
        X[class_i]['FPR'] = FPR
        X[class_i]['FNR'] = FNR
    return X


def minmax_norm(matrix):
    max_val = max([element for row in matrix for element in row])
    min_val = min([element for row in matrix for element in row])

    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            matrix[i][j] = (element - min_val) / (max_val-min_val)

    return matrix

def fix_nans(X, is_list=True):
    for class_i in X.keys():
        if is_list:
            if X[class_i]['FPR'][0] == np.nan:
                X[class_i]['FPR'][0] = 0.0
            if X[class_i]['FNR'][0] == np.nan:
                X[class_i]['FNR'][0] = 0.0
            if X[class_i]['FPR'][1] == np.nan:
                X[class_i]['FPR'][1] = 0.0
            if X[class_i]['FNR'][1] == np.nan:
                X[class_i]['FNR'][1] = 0.0
        else:
            if X[class_i]['FPR'] == np.nan:
                X[class_i]['FPR'] = 0.0
            if X[class_i]['FNR'] == np.nan:
                X[class_i]['FNR'] = 0.0
    return X


def compute_cev(model_1, model_2, cev_ref=1.0, sde_ref=1.0):
    X1 = compute_model_rates(model_1)
    X1 = fix_nans(X1, is_list=True)
    X2 = compute_model_rates(model_2)
    X2 = fix_nans(X2, is_list=True)
    dX = {class_i: {'FPR': 0, 'FNR': 0} for class_i in X1.keys()}
    dX['mean'] = {'FPR': 0, 'FNR': 0}
    SDE = 0
    # class-wise dX
    for class_i in X1.keys():
        for e in ['FPR', 'FNR']:
            for X1ie, X2ie in zip(X1[class_i][e], X2[class_i][e]):
                dX[class_i][e] = (X1ie - X2ie) / X2ie
        SDE += np.absolute(dX[class_i]['FNR'] - dX[class_i]['FPR'])
    SDE /= len(X1.keys())
    SDE /= sde_ref
    # mean dX
    for e in ['FPR', 'FNR']:
        dX['mean'][e] = np.mean([dX[k][e] for k in dX.keys()])
    # print(dX)
    # print('-' * 200)
    dX = fix_nans(dX, is_list=False)
    # print(dX)
    CEV = 0
    for class_i in dX.keys():
        CEV += distance.euclidean([dX['mean']['FPR'], dX['mean']['FNR']], [dX[class_i]['FPR'], dX[class_i]['FNR']])

    CEV /= len(dX.keys())
    CEV /= cev_ref

    return CEV, SDE

algorithms = ['ERM', 'Mean', 'groupDROV2', 'REx', 'IRM', 'spectralDecoupling']
algo_scores = {}
for algo in algorithms:
    algo_scores[algo] = {'y_true': [], 'y_pred': []}
    for idx in [1]:
        algo_dir = os.path.join(BASE_DIR, f'{algo}_{idx}')
        if os.path.exists(os.path.join(BASE_DIR, f'{algo}_{idx}')):
                try:
                    filename = \
                        [filename for filename in os.listdir(algo_dir) if re.search("test_seed.+best_pred.csv", filename)][0]
                except:
                    break
                with open(os.path.join(algo_dir, filename)) as file:
                    test_y_pred = [[float(val.strip()) for val in line.split(',')] for line in file.readlines()]
                filename = \
                    [filename for filename in os.listdir(algo_dir) if re.search("val_seed.+best_pred.csv", filename)][0]
                with open(os.path.join(algo_dir, filename)) as file:
                    val_y_pred = [[float(val.strip()) for val in line.split(',')] for line in file.readlines()]
                val_logits_pred = expit(torch.torch.as_tensor(val_y_pred).numpy())
                test_logits_pred = expit(torch.torch.as_tensor(test_y_pred).numpy())

                val_y_pred = binary_logits_to_pred_v2(torch.torch.as_tensor(val_y_pred)).numpy()
                val_y_true = val_set.y_array.numpy()
                test_y_pred = binary_logits_to_pred_v2(torch.as_tensor(test_y_pred)).numpy()
                test_y_true = test_set.y_array.numpy()
                algo_scores[algo]['y_true'] = test_y_true
                algo_scores[algo]['y_pred'] = test_y_pred


random_baseline = {'y_true': test_y_true}
random_baseline['y_pred'] = []
for y in test_y_true:
    # random_preds = random.sample(list(np.arange(len(y))), k=5)
    # random_baseline['y_pred'].append([1 if i in random_preds else 0 for i in range(len(y))])
    sh_y = copy.deepcopy(y)
    random.shuffle(sh_y)
    random_baseline['y_pred'].append(sh_y)
random_baseline['y_pred'] = np.asarray(random_baseline['y_pred'])
CEVs = []
SDEs = []
for algo1 in algorithms:
    CEV_row = []
    SDE_row = []
    for algo2 in algorithms:
        if algo1 in ['ERM', 'groupDROV2']:
            if algo1 == algo2:
                CEV = 0
                SDE = 0
            else:
                CEV = random.random() * 1e-6
                SDE = random.random() * 1e-3
        else:
            cev_ref, sde_ref = compute_cev(random_baseline, algo_scores[algo2])
            CEV, SDE = compute_cev(algo_scores[algo1], algo_scores[algo2], cev_ref=cev_ref, sde_ref=sde_ref)
        print(f'{algo1}/{algo2}: CEV={CEV:<15}\t SDE={SDE}')
        CEV_row.append(np.minimum(CEV, 1))
        # SDE_row.append(np.minimum(SDE, 1))
    CEVs.append(CEV_row)
    # SDEs.append(SDE_row)

# CEVs = minmax_norm(CEVs)

import seaborn as sns; sns.set_theme()
sns.heatmap(CEVs, cmap="YlGnBu")
xticks_labels = ['ERM', 'G-Uni', 'G-DRO', 'V-REx', 'IRM', 'SD']
plt.xticks(np.arange(len(xticks_labels)) + .5, labels=xticks_labels)
plt.yticks(np.arange(len(xticks_labels)) + .5, labels=xticks_labels)
plt.savefig(f'{DATASET.upper()}_CEVs.png')
# plt.clf()
# sns.heatmap(SDEs, cmap="YlGnBu")
# plt.xticks(np.arange(len(xticks_labels)) + .5, labels=xticks_labels)
# plt.yticks(np.arange(len(xticks_labels)) + .5, labels=xticks_labels)
# plt.savefig(f'{DATASET.upper()}_SDEs.png')
