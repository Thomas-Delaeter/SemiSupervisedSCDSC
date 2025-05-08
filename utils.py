from __future__ import division, print_function
import numpy as np
import torch
from collections import Counter
from munkres import Munkres
from sklearn.metrics import accuracy_score
from sklearn import metrics

#######################################################
# Evaluate Critiron
#######################################################
def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    ca = class_acc(y_true, y_best)

    # Map of predicted clusters to true clusters
    cluster_mapping = {int(Label2[i]): int(Label1[c[i]]) for i in range(len(Label2))}

    if return_aligned:
        return y_best, acc, kappa, nmi, ca, cluster_mapping
    return acc, kappa, nmi


def class_acc(y_true, y_pre):
    """
    calculate each classes's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca

def ascii_histogram(labels, max_bar_width=50):
    distribution = Counter(labels)
    max_count = max(distribution.values())

    print("Class Distribution Histogram (ASCII)")
    for label, count in sorted(distribution.items()):
        bar_length = int(count / max_count * max_bar_width)
        bar = '#' * bar_length  # You can change to '#' or other characters if preferred
        print(f"{str(label):>5}: {bar} ({count})")

def get_labeled_data(y, X, label_size = 3, random_state = 42):
    """
    Splits X, y into labeled and unlabeled sets.
    If 0 < label_size < 1: treats as fraction of total samples.
    If label_size >= 1: treats as number of labels *per class*, and
        uses a fixed random permutation per class so that increasing
        label_size simply extends the prefix of that permutation.
    """

    if label_size < 1:
        # passed as decimal number: 0.01 = 1% labeled data used
        label_pct = label_size

        # TODO: chances are that not all classes are within the l_feats selection
        # get random indices throughout the entire dataset
        indices = torch.randperm(len(y))
        train_size = int(label_pct * len(y))  # only use a percentage of all the indices
        train_indices = indices[:train_size]

        # Create a boolean mask of the same length as y, initialized to False
        mask_lab = torch.zeros(len(y),dtype=torch.bool)
        # Set the indices corresponding to labelled samples to True
        mask_lab[train_indices] = True

        u_feats = X[~mask_lab]
        # u_targets = y[~mask_lab]
        l_feats = X[mask_lab]
        l_targets = y[mask_lab]

        return u_feats, l_feats, l_targets, mask_lab

    else:

        rng = np.random.RandomState(random_state)

        # fixed number per class, but incremental via prefix of a permuted list
        num_labeled_per_class = int(label_size)

        unique_classes = np.unique(y)
        mask_lab = np.zeros(len(y), dtype=bool)

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]

            if len(cls_indices) < num_labeled_per_class:
                raise ValueError(
                    f"Class {cls} has only {len(cls_indices)} samples, "
                    f"but {num_labeled_per_class} were requested."
                )

            # numpy permutation of that class’s indices
            perm = rng.permutation(cls_indices)
            selected = perm[:num_labeled_per_class]
            mask_lab[selected] = True

        # If X is a torch.Tensor, convert the mask to torch; otherwise numpy works
        if isinstance(X, torch.Tensor):
            torch_mask = torch.from_numpy(mask_lab)
            l_feats = X[torch_mask]
            u_feats = X[~torch_mask]
        else:
            l_feats = X[mask_lab]
            u_feats = X[~mask_lab]

        l_targets = y[mask_lab]

        return u_feats, l_feats, l_targets, mask_lab

import torch
import numpy as np

def get_labeled_data_strat(y, X, label_size=3, random_state=42):
    """
    Splits X, y into labeled and unlabeled sets.
    If 0 < label_size < 1: percentage‐based random sampling.
    If label_size >= 1: draws 'label_size' samples per class by:
       1) sorting that class’s indices in dataset order
       2) splitting them into label_size bins
       3) picking one random index from each bin
    """
    # one RNG for this call
    rng = np.random.RandomState(random_state)

    if label_size < 1:
        # ... your existing percentage branch unchanged ...
        label_pct = label_size
        indices = rng.permutation(len(y))
        train_size = int(label_pct * len(y))
        train_indices = indices[:train_size]

        mask_lab = torch.zeros(len(y), dtype=torch.bool)
        mask_lab[train_indices] = True

        u_feats = X[~mask_lab]
        l_feats = X[mask_lab]
        l_targets = y[mask_lab]
        return u_feats, l_feats, l_targets, mask_lab

    else:
        # stratified *within* each class
        num_per_class = int(label_size)
        unique_classes = np.unique(y)
        mask_lab = np.zeros(len(y), dtype=bool)

        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) < num_per_class:
                raise ValueError(
                    f"Class {cls} has only {len(cls_idx)} samples, "
                    f"but {num_per_class} were requested."
                )

            # sort by dataset order to define strata
            cls_idx_sorted = np.sort(cls_idx)

            # split into `num_per_class` bins (as even as possible)
            strata = np.array_split(cls_idx_sorted, num_per_class)

            # pick one random index from each stratum
            picks = [int(rng.choice(bin_)) for bin_ in strata if len(bin_) > 0]
            mask_lab[picks] = True

        # build your labeled / unlabeled sets
        if isinstance(X, torch.Tensor):
            tm = torch.from_numpy(mask_lab)
            l_feats, u_feats = X[tm], X[~tm]
        else:
            l_feats, u_feats = X[mask_lab], X[~mask_lab]

        l_targets = y[mask_lab]
        return u_feats, l_feats, l_targets, mask_lab




