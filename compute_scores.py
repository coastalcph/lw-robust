import os
import re

import numpy as np
import torch

from dataloaders import get_dataset
from configs.supported import algorithms
from configs.supported import F1, binary_logits_to_pred_v2
from metrics.retrieval_metrics import mean_rprecision
from scipy.special import expit


DATASETS = 'eurlex100'
BASE_DIR = f'/Users/rwg642/Desktop/MULTI-LABEL-ROBUSTNESS/RESULTS/{DATASET}/BATCH_64_SMALL_MORE_GROUPS'
SPLIT_SCHEME = 'official'
from sklearn.metrics import f1_score

full_dataset = get_dataset(
    dataset=DATASET,
    version='2.0',
    root_dir='data/datasets',
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
val_reports = []
test_reports = []
score_dicts = {}
done_algos = []
for algo in algorithms:
    score_dict = {'dev': {'micro': [], 'macro': [], 'rp': []},
                  'dev_few': {'micro': [], 'macro': [], 'rp': []},
                  'dev_freq': {'micro': [], 'macro': [], 'rp': []},
                  'test': {'micro': [], 'macro': [], 'rp': []},
                  'test_few': {'micro': [], 'macro': [], 'rp': []},
                  'test_freq': {'micro': [], 'macro': [], 'rp': []}}

    for idx in [1,2,3]:
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
                val_f1 = f1_score(val_y_true, val_y_pred, average='micro', zero_division=0)
                # if val_f1 < 0.6:
                #     continue
                score_dict['dev']['micro'].append(f1_score(val_y_true, val_y_pred, average='micro', zero_division=0))
                score_dict['dev']['macro'].append(f1_score(val_y_true, val_y_pred, average='macro', zero_division=0))
                score_dict['dev']['rp'].append(mean_rprecision(val_y_true, val_logits_pred)[0])
                score_dict['test']['micro'].append(f1_score(test_y_true, test_y_pred, average='micro', zero_division=0))
                score_dict['test']['macro'].append(f1_score(test_y_true, test_y_pred, average='macro', zero_division=0))
                score_dict['test']['rp'].append(mean_rprecision(test_y_true, test_logits_pred)[0])

                # FREQ-SHOT
                label_counts = np.sum(train_y_true, axis=0)
                limit = sorted(label_counts)[int(len(label_counts) * 0.5)]
                # print(label_counts.shape)
                indices = np.expand_dims(np.asarray([label_idx for label_idx, label_count in enumerate(label_counts)
                                                     if label_count >= limit and label_count != 0], dtype=np.int32),
                                         axis=0)
                val_indices = indices.repeat(len(val_y_true), axis=0)
                test_indices = indices.repeat(len(test_y_true), axis=0)
                # print(val_indices.shape)
                # print(val_y_true.shape)
                # print(test_indices.shape)
                # print(test_y_true.shape)
                # exit()
                freq_val_y_true = np.take_along_axis(val_y_true, val_indices, axis=1)
                freq_val_y_pred = np.take_along_axis(val_y_pred, val_indices, axis=1)
                freq_val_logits_pred = np.take_along_axis(val_logits_pred, val_indices, axis=1)
                freq_test_y_true = np.take_along_axis(test_y_true, test_indices, axis=1)
                freq_test_y_pred = np.take_along_axis(test_y_pred, test_indices, axis=1)
                freq_test_logits_pred = np.take_along_axis(test_logits_pred, test_indices, axis=1)
                score_dict['dev_freq']['micro'].append(
                    f1_score(freq_val_y_true, freq_val_y_pred, average='micro', zero_division=0))
                score_dict['dev_freq']['macro'].append(
                    f1_score(freq_val_y_true, freq_val_y_pred, average='macro', zero_division=0))
                score_dict['dev_freq']['rp'].append(mean_rprecision(freq_val_y_true, freq_val_logits_pred)[0])
                score_dict['test_freq']['micro'].append(
                    f1_score(freq_test_y_true, freq_test_y_pred, average='micro', zero_division=0))
                score_dict['test_freq']['macro'].append(
                    f1_score(freq_test_y_true, freq_test_y_pred, average='macro', zero_division=0))
                score_dict['test_freq']['rp'].append(mean_rprecision(freq_test_y_true, freq_test_logits_pred)[0])

                # FEW-SHOT
                label_counts = np.sum(train_y_true, axis=0)
                limit = sorted(label_counts)[int(len(label_counts)*0.5)]
                # print(label_counts.shape)
                indices = np.expand_dims(np.asarray([label_idx for label_idx, label_count in enumerate(label_counts)
                           if label_count < limit and label_count != 0], dtype=np.int32), axis=0)
                val_indices = indices.repeat(len(val_y_true), axis=0)
                test_indices = indices.repeat(len(test_y_true), axis=0)
                # print(val_indices.shape)
                # print(val_y_true.shape)
                # print(test_indices.shape)
                # print(test_y_true.shape)
                # exit()
                few_val_y_true = np.take_along_axis(val_y_true, val_indices, axis=1)
                few_val_y_pred = np.take_along_axis(val_y_pred, val_indices, axis=1)
                few_val_logits_pred = np.take_along_axis(val_logits_pred, val_indices, axis=1)
                few_test_y_true = np.take_along_axis(test_y_true, test_indices, axis=1)
                few_test_y_pred = np.take_along_axis(test_y_pred, test_indices, axis=1)
                few_test_logits_pred = np.take_along_axis(test_logits_pred, test_indices, axis=1)
                score_dict['dev_few']['micro'].append(f1_score(few_val_y_true, few_val_y_pred, average='micro', zero_division=0))
                score_dict['dev_few']['macro'].append(f1_score(few_val_y_true, few_val_y_pred, average='macro', zero_division=0))
                score_dict['dev_few']['rp'].append(mean_rprecision(few_val_y_true, few_val_logits_pred)[0])
                score_dict['test_few']['micro'].append(f1_score(few_test_y_true, few_test_y_pred, average='micro', zero_division=0))
                score_dict['test_few']['macro'].append(f1_score(few_test_y_true, few_test_y_pred, average='macro', zero_division=0))
                score_dict['test_few']['rp'].append(mean_rprecision(few_test_y_true, few_test_logits_pred)[0])

        else:
            break
    if len(score_dict['dev']['micro']):
        score_dicts[algo] = score_dict

print('-' * 200)
print('ALL LABELS')
print('-' * 200)

print(f'{" "*26} {"VALIDATION":<70} | {"TEST":<70}')
print('-' * 200)
for algo, stats in score_dicts.items():
    print(f'{algo:>25}: MICRO-F1: {np.mean(stats["dev"]["micro"])*100:.1f} ± {np.std(stats["dev"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["dev"]["macro"])*100:.1f} ± {np.std(stats["dev"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["dev"]["rp"])*100:.1f} ± {np.std(stats["dev"]["rp"])*100:.1f} | '
          f'MICRO-F1: {np.mean(stats["test"]["micro"])*100:.1f} ± {np.std(stats["test"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["test"]["macro"])*100:.1f} ± {np.std(stats["test"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["test"]["rp"])*100:.1f} ± {np.std(stats["test"]["rp"])*100:.1f}')


print('-' * 200)
print('FREQUENT LABELS')
print('-' * 200)

print(f'{" "*26} {"VALIDATION":<70} | {"TEST":<70}')
print('-' * 200)
for algo, stats in score_dicts.items():
    print(f'{algo:>25}: MICRO-F1: {np.mean(stats["dev_freq"]["micro"])*100:.1f} ± {np.std(stats["dev_freq"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["dev_freq"]["macro"])*100:.1f} ± {np.std(stats["dev_freq"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["dev_freq"]["rp"])*100:.1f} ± {np.std(stats["dev_freq"]["rp"])*100:.1f} | '
          f'MICRO-F1: {np.mean(stats["test_freq"]["micro"])*100:.1f} ± {np.std(stats["test_freq"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["test_freq"]["macro"])*100:.1f} ± {np.std(stats["test_freq"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["test_freq"]["rp"])*100:.1f} ± {np.std(stats["test_freq"]["rp"])*100:.1f}')

print('-' * 200)
print('FEW LABELS')
print('-' * 200)

print(f'{" "*26} {"VALIDATION":<70} | {"TEST":<70}')
print('-' * 200)
for algo, stats in score_dicts.items():
    print(f'{algo:>25}: MICRO-F1: {np.mean(stats["dev_few"]["micro"])*100:.1f} ± {np.std(stats["dev_few"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["dev_few"]["macro"])*100:.1f} ± {np.std(stats["dev_few"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["dev_few"]["rp"])*100:.1f} ± {np.std(stats["dev_few"]["rp"])*100:.1f} | '
          f'MICRO-F1: {np.mean(stats["test_few"]["micro"])*100:.1f} ± {np.std(stats["test_few"]["micro"])*100:.1f}\t'
          f'MACRO-F1: {np.mean(stats["test_few"]["macro"])*100:.1f} ± {np.std(stats["test_few"]["macro"])*100:.1f}\t'
          f'R-PRECISION: {np.mean(stats["test_few"]["rp"])*100:.1f} ± {np.std(stats["test_few"]["rp"])*100:.1f}')


