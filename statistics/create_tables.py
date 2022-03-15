import os
import re

import numpy as np
import torch

from dataloaders import get_dataset
from configs.supported import algorithms
from configs.supported import F1, binary_logits_to_pred_v2
from metrics.retrieval_metrics import mean_rprecision
from scipy.special import expit


DATASETS = [('uk_legislation', 1.0), ('uk_legislation69', 2.0), ('eurlex', 1.0),
            ('eurlex100', 2.0), ('bioasq16', 1.0), ('bioasq116', 2.0)]

# DATASETS = [('uk_legislation69', 2.0), ('eurlex100', 2.0),  ('bioasq116', 2.0)]
#
# DATASETS = [('mimic20', 1.0), ('mimic200', 2.0)]

# DATASETS = [('eurlex100', 2.0)]

DATASETS = [('eurlex', 1.0), ('eurlex100', 2.0),  ('eurlex500', 3.0)]



score_datasets = {}
for DATASET, VERSION in DATASETS:
    BASE_DIR = f'/Users/rwg642/Desktop/MULTI-LABEL-ROBUSTNESS/FINAL_RESULTS_LWAN/{DATASET}/'
    SPLIT_SCHEME = 'official'
    from sklearn.metrics import f1_score

    full_dataset = get_dataset(
        dataset=DATASET,
        version=VERSION,
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
    val_reports = []
    test_reports = []
    score_dicts = {}
    done_algos = []
    for algo in ['ERM',  'IRM','spectralDecoupling', 'BMOV3']:
        score_dict = {'dev': {'micro': [], 'macro': [], 'rp': []},
                      'dev_few': {'micro': [], 'macro': [], 'rp': []},
                      'dev_freq': {'micro': [], 'macro': [], 'rp': []},
                      'test': {'micro': [], 'macro': [], 'rp': []},
                      'test_few': {'micro': [], 'macro': [], 'rp': []},
                      'test_freq': {'micro': [], 'macro': [], 'rp': []}}
        for idx in [1, 2, 3]:
            try:
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

                        score_dict['dev']['micro'].append(f1_score(val_y_true, val_y_pred, average='micro', zero_division=0))
                        score_dict['dev']['macro'].append(f1_score(val_y_true, val_y_pred, average='macro', zero_division=0))
                        score_dict['dev']['rp'].append(mean_rprecision(val_y_true, val_logits_pred)[0])
                        score_dict['test']['micro'].append(f1_score(test_y_true, test_y_pred, average='micro', zero_division=0))
                        score_dict['test']['macro'].append(f1_score(test_y_true, test_y_pred, average='macro', zero_division=0))
                        score_dict['test']['rp'].append(mean_rprecision(test_y_true, test_logits_pred)[0])

                        # FREQ-SHOT
                        label_counts = np.sum(train_y_true, axis=0)
                        limit = sorted(label_counts)[int(len(label_counts) * 0.5)]
                        indices = np.expand_dims(np.asarray([label_idx for label_idx, label_count in enumerate(label_counts)
                                                             if label_count >= limit and label_count != 0], dtype=np.int32),
                                                 axis=0)
                        val_indices = indices.repeat(len(val_y_true), axis=0)
                        test_indices = indices.repeat(len(test_y_true), axis=0)

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
                        indices = np.expand_dims(np.asarray([label_idx for label_idx, label_count in enumerate(label_counts)
                                   if label_count < limit and label_count != 0], dtype=np.int32), axis=0)
                        val_indices = indices.repeat(len(val_y_true), axis=0)
                        test_indices = indices.repeat(len(test_y_true), axis=0)

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
            except:
                continue

        # KEEP BEST SCORES
        seed_scores = [(idx, score) for (idx, score) in enumerate(score_dict['dev']['rp'])]
        sorted_scores = sorted(seed_scores, key=lambda tup: tup[1], reverse=True)
        top_k_ids = [idx for idx, score in sorted_scores[:1]]
        for subset in ['dev', 'test', 'dev_freq', 'dev_few', 'test_freq', 'test_few']:
            for metric in ['micro', 'macro', 'rp']:
                try:
                    score_dict[subset][metric] = (score_dict[subset][metric][top_k_ids[0]], np.mean(score_dict[subset][metric]), np.std(score_dict[subset][metric]))
                except:
                    if isinstance(score_dict[subset][metric], list):
                        if len(score_dict[subset][metric]):
                            score_dict[subset][metric] = (score_dict[subset][metric][0], np.mean(score_dict[subset][metric]), np.std(score_dict[subset][metric]))
                        else:
                            score_dict[subset][metric] = (0.0, 0.0, 0.0)
        score_dicts[algo] = score_dict

    score_datasets[DATASET] = score_dicts


# for algo, algo_name in zip(['ERM', 'ERMGS', 'Mean', 'groupDRO', 'deepCORAL', 'IRM', 'REx', 'spectralDecoupling', 'BMOV2','BMOV3'],
#                            ['ERM', 'ERM+GS', 'Group Uniform', 'Group DRO', 'Deep CORAL', 'IRM', 'REx', 'SD', 'LW-DRO (v1)', 'LW-DRO (v2)']):
#     algo_report = f'{algo_name:<15} & '
#     for DATASET, VERSION in DATASETS:
#         for subset in ['test', 'test_freq', 'test_few']:
#             for metric in ['micro', 'macro', 'rp']:
#                 diff = int(score_datasets[DATASET][algo][subset][metric]*100 - score_datasets[DATASET]['ERM'][subset][metric] * 100)
#                 if diff > 0:
#                     color_code = '\cellcolor{c' + str(diff) + '} '
#                 elif diff < 0:
#                     color_code = '\cellcolor{n' + str(abs(diff)) + '} '
#                 else:
#                     color_code = ''
#                 algo_report += f'{color_code}{score_datasets[DATASET][algo][subset][metric]*100:<.1f} & ' \
#                     if score_datasets[DATASET][algo][subset][metric] != 0 else '00.0 & '
#                 algo_report += ' '
#     algo_report = algo_report[:-2] + '\\\\ '
#     print(algo_report)




for algo, algo_name in zip(['ERM', 'IRM', 'spectralDecoupling', 'BMOV3'],
                           ['ERM', 'IRM', 'SD', 'LW-DRO']):
    algo_report = f'{algo_name:<15} & '
    for DATASET, VERSION in DATASETS:
        for subset in ['test']:
            for metric in ['micro', 'macro', 'rp']:
                algo_report += f'{score_datasets[DATASET][algo][subset][metric][0]*100:<.1f} & ' \
                    if score_datasets[DATASET][algo][subset][metric][0] != 0 else '00.0 & '
                algo_report += ' '
    algo_report = algo_report[:-2] + '\\\\ '
    print(algo_report)

# for algo, algo_name in zip(['ERM', 'ERMGS', 'Mean', 'groupDRO', 'deepCORAL', 'REx', 'IRM', 'spectralDecoupling'],
#                            ['ERM', 'ERM+GS', 'Group Uniform', 'Group DRO', 'Deep CORAL', 'V-REx', 'IRM', 'SD']):
#     algo_report = f'{algo_name:<15} & '
#     for DATASET, VERSION in DATASETS:
#         for subset in ['test', 'test_freq', 'test_few']:
#             for metric in ['micro', 'macro', 'rp']:
#                 diff = int(score_datasets[DATASET][algo][subset][metric]*100 - score_datasets[DATASET]['ERM'][subset][metric] * 100)
#                 if diff > 0:
#                     color_code = '\cellcolor{c' + str(diff) + '} '
#                 elif diff < 0:
#                     color_code = '\cellcolor{n' + str(abs(diff)) + '} '
#                 else:
#                     color_code = ''
#                 algo_report += f'{color_code}{score_datasets[DATASET][algo][subset][metric]*100:<.1f} & ' \
#                     if score_datasets[DATASET][algo][subset][metric] != 0 else '00.0 & '
#                 algo_report += ' '
#     algo_report = algo_report[:-2] + '\\\\ '
#     print(algo_report)

#
# for algo, algo_name in zip(['ERM', 'ERMGS', 'Mean', 'groupDRO', 'deepCORAL', 'IRM', 'REx', 'spectralDecoupling'],
#                            ['ERM', 'ERM+GS', 'Group Uniform', 'Group DRO', 'Deep CORAL', 'IRM', 'REx', 'SD']):
#     algo_report = f'{algo_name:<15} & '
#     for DATASET, VERSION in DATASETS:
#         for metric in ['micro', 'macro', 'rp']:
#             diff = int(score_datasets[DATASET][algo][subset][metric]*100 - score_datasets[DATASET]['ERM'][subset][metric] * 100)
#             if diff > 0:
#                 color_code = '\cellcolor{c' + str(diff) + '} '
#             elif diff < 0:
#                 color_code = '\cellcolor{n' + str(abs(diff)) + '} '
#             else:
#                 color_code = ''
#             algo_report += f'{color_code}{score_datasets[DATASET][algo][subset][metric]*100:<.1f} & ' \
#                 if score_datasets[DATASET][algo][subset][metric] != 0 else '00.0 & '
#             algo_report += ' '
#     algo_report = re.sub('& $', '\\\\ ', algo_report)
#     print(algo_report)

# subset = 'dev'
# for metric in ['micro', 'macro', 'rp']:
#     for algo, algo_name in zip(
#             ['ERM', 'ERMGS', 'Mean', 'groupDRO', 'deepCORAL', 'IRM', 'REx', 'spectralDecoupling'],
#             ['ERM', 'ERM+GS', 'Group Uniform', 'Group DRO', 'Deep CORAL', 'IRM', 'REx', 'SD']):
#         algo_report = f'{algo_name:<15} & '
#         for DATASET, VERSION in DATASETS:
#             algo_report += f'{score_datasets[DATASET][algo][subset][metric][1]*100:<.1f} $\pm$ {score_datasets[DATASET][algo][subset][metric][2]*100:<.1f} & ' \
#                 if score_datasets[DATASET][algo][subset][metric] != 0 else '00.0 $\pm$ 00.0 & '
#             algo_report += ' '
#         algo_report = re.sub('& $', '\\\\ ', algo_report)
#         print(algo_report)
#     print('-'*200)