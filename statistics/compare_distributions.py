from dataloaders import get_dataset
import numpy as np
from math import log2
from scipy.stats import wasserstein_distance

# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) if q[i] !=0 else p[i] * log2(p[i]/q[i] + 1e-8) for i in range(len(p)))


for dataset, version in [('uk_legislation', '1.0'), ('uk_legislation69', '3.0'),
                         ('eurlex', '1.0'), ('eurlex100', '2.0'),
                         ('bioasq16', '1.0'), ('bioasq116', '2.0')]:
    print('-' * 100)
    for split in ['shuffled', 'official']:
        full_dataset = get_dataset(
            dataset=dataset,
            version=version,
            root_dir='../data/datasets',
            split_scheme=split)

        label_dists = []
        train_set = full_dataset.get_subset('train')
        y_true = train_set.y_array.numpy()
        train_label_dist = np.sum(y_true, axis=0) / len(y_true)

        for subset in ['val', 'test']:
            sub_set = full_dataset.get_subset(subset)
            y_true = sub_set.y_array.numpy()
            label_dist = np.sum(y_true, axis=0) / len(y_true)
            print(f'{dataset.upper():>15}\t ({subset})\t ({split:>10})\t KL-Divergence: {kl_divergence(train_label_dist, label_dist):3f}\t Wasserstein Distance; {wasserstein_distance(train_label_dist,label_dist):.3f}')
