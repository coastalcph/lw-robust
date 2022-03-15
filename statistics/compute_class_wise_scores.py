import os
import re

import numpy as np
import torch

from dataloaders import get_dataset
from configs.supported import binary_logits_to_pred_v2
from scipy.special import expit
DATASET = 'eurlex100'
BASE_DIR = f'/home/iliasc/temporal-wilds/logs/{DATASET}/BATCH_64_SMALL_MORE_GROUPS'
SPLIT_SCHEME = 'official'
from sklearn.metrics import f1_score

full_dataset = get_dataset(
    dataset=DATASET,
    version='1.0',
    root_dir='../data/datasets',
    split_scheme=SPLIT_SCHEME)

train_set = full_dataset.get_subset('train')
train_y_true = train_set.y_array.numpy()
print(len(train_y_true))

test_set = full_dataset.get_subset('test')
test_y_true = test_set.y_array.numpy()
print(len(test_y_true))

val_set = full_dataset.get_subset('val')
val_y_true = test_set.y_array.numpy()
print(len(val_y_true))

SPLITS = ['train', 'dev', 'test']
score_dicts = {}
for algo in ['ERM', 'IRM', 'spectralDecoupling']:
    score_dicts[algo] = []
    for idx in range(1, 4):
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
                score_dicts[algo].append(f1_score(test_y_true, test_y_pred, zero_division=0, average=None))

        else:
            break

    score_dicts[algo] = np.asarray([np.asarray(run) for run in score_dicts[algo]])

label_counts = np.sum(train_y_true, axis=0)
label_indices = [(idx, count) for idx, count in zip(np.arange(len(label_counts)), label_counts)]
label_indices.sort(key=lambda x: x[1])
label_indices = np.asarray([idx for idx, _ in label_indices], dtype=np.int32)
label_indices = np.flip(label_indices)

erm_scores = np.take_along_axis(np.mean(score_dicts['ERM'], axis=0), label_indices, axis=0)
irm_scores = np.take_along_axis(np.mean(score_dicts['IRM'], axis=0), label_indices, axis=0)
sd_scores = np.take_along_axis(np.mean(score_dicts['spectralDecoupling'], axis=0), label_indices, axis=0)

top_scores = [max([x,y,z]) for x,y,z in zip(erm_scores, irm_scores, sd_scores)]
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
x_pos = np.arange(0, len(label_indices)*40, step=40)
x_pos_2 = [x+10 for x in x_pos]
x_pos_3 = [x+20 for x in x_pos]
print(len(x_pos))
print(len(x_pos_2))
# Build the plot
fig, ax = plt.subplots(figsize=(40, 10))
plt.bar(x_pos_2, top_scores, align='center', alpha=0.1, ecolor='black', capsize=100, color='black', width=30)
plt.bar(x_pos, erm_scores, align='center', alpha=0.6, ecolor='black', capsize=100, color='blue', label='ERM', width=10)
plt.bar(x_pos_2, irm_scores, align='center', alpha=0.6, ecolor='black', capsize=100, color='orange', label='IRM', width=10)
plt.bar(x_pos_3, sd_scores, align='center', alpha=0.6, ecolor='black', capsize=100, color='red', label='SD', width=10)
# few_line = x_pos_2[int(len(x_pos_2)/2)-1]
# ax.plot([few_line+10, few_line+10], [0, 1],  linewidth=3, color='black', linestyle='dashed')
ax.set_xticks(x_pos_2)
ax.set_xticklabels(np.arange(1, len(label_indices)+1))
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(np.arange(10, 110, step=10))
ax.yaxis.grid(True)
plt.legend(loc='upper right')
# Save the figure and show
plt.tight_layout()
plt.savefig('CLASS-WISE-SCORES_EURLEX100.png', dpi=300)
# plt.show()







