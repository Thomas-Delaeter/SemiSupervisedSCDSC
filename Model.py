from __future__ import print_function, division
import argparse
import shutil
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from consistency_regularization import WeakAug, HSIRandAugment
from consistency_regularization_gpu import WeakAug as WeakAugGPU
from consistency_regularization_gpu import HSIRandAugment as HSIRandAugGPU
from consistency_regularization_latent import WeakLatentAug, StrongLatentAug


from my_knn import get_initial_value
from Auto_encoder import Ae
from getdata import Load_my_Dataset
from utils import cluster_accuracy, ascii_histogram, get_labeled_data, get_labeled_data_strat
import warnings
from Initialize_D import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
import os
import random
import optuna

from faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from cluster_and_log_utils import log_accs_from_preds
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, cohen_kappa_score
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    labels = np.array(labels).flatten()  # ensure it's 1D

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels  # elementwise boolean array

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(in_bin):
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / len(probs)) * np.abs(bin_accuracy - bin_confidence)

    return ece

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        os.listdir(filepath)
        print(os.listdir(filepath))

warnings.filterwarnings("ignore")
epochs = 400

class C_EDESC(nn.Module):
    def __init__(self,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain_path):
        super(C_EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = Ae(
            n_input=n_input,
            n_z=n_z)

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z * 7 * 7, n_clusters))
        nn.init.xavier_uniform_(self.D) # stable inplace init of D
        print(self.D.shape)

        # augmentation functions called in foward pass
        # augmentations did not help multi-view learning, meaning that they already provide distinct feature learning
        self.weak_latent_aug = WeakLatentAug(noise_std=0.01, flip_prob=0.5)
        self.strong_latent_aug = StrongLatentAug(
            noise_std=0.1,
            drop_prob=0.2,
            shift=True,
            flip=True
        )

        # learnable weights of the loss func, uncertainty-weight inspired (not used)
        # cross, pseudo, contrastive, fixmatch
        self.log_vars_semisup = nn.Parameter(torch.tensor([args.delta, args.lmdb, args.omega, args.psi]))

        # Pseudo-Label Module
        self.pseudo_classifier = nn.Sequential(
            nn.Conv2d(n_z, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, n_clusters)
        )

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
            self.ae.load_state_dict(torch.load(self.pretrain_path,map_location=device))
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path,map_location=device))
        print('Load pre-trained model from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)

        # z_weak = self.weak_latent_aug(z)
        # z_strong = self.strong_latent_aug(z)

        # Pseudo Label
        pseudo_logits = self.pseudo_classifier(z)
        pseudo_logits_aug = None#= self.pseudo_classifier(z_strong)

        z_shape = z.shape
        num_ = z_shape[0]
        z = z.reshape((num_, -1))
        d = args.d
        s = None
        eta = args.eta

        # Calculate subspace affinity
        for i in range(self.n_clusters):
            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + eta * d) / ((eta + 1) * d)
        s = (s.t() / torch.sum(s, 1)).t()

        s_weak = None
        # z_weak = z_weak.reshape((num_, -1))
        # for i in range(self.n_clusters):
        #     si = torch.sum(torch.pow(torch.mm(z_weak, self.D[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
        #     if s_weak is None:
        #         s_weak = si
        #     else:
        #         s_weak = torch.cat((s_weak, si), 1)
        # s_weak = (s_weak + eta * d) / ((eta + 1) * d)
        # s_weak = (s_weak.t() / torch.sum(s_weak, 1)).t()


        # Augmenting the soft assignment matrix or latent environment did not provide meaningful contributions
        return x_bar, (s, s_weak), z, (pseudo_logits, pseudo_logits_aug)

    def total_loss(self, x, x_bar, center, target, dim, n_clusters, s, index):
        # Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)
        kl_loss = F.kl_div(center.log(), target.data)
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)

        location = torch.tensor(index, device=device).T
        smooth_pred = spatial_filter(s, location, args.image_size)
        loss_smooth = F.kl_div(s.log(), smooth_pred.data)

        total_loss = reconstr_loss + loss_d1 + loss_d2 + args.alpha * kl_loss + args.beta * loss_smooth

        return total_loss, reconstr_loss, kl_loss, loss_d1, loss_d2

    def pseudo_loss(self, y, pseudo_logits):

        pseudo_loss = torch.tensor(0.0, device=device)
        if args.lmdb == .0:
            return pseudo_loss

        print(f"Pseudo accuracy was {accuracy_score(torch.argmax(pseudo_logits.cpu(), dim=1), y)}")

        # Samples with high confidence (e.g. max probability > 0.8 (args.pseudo_threshold), like PSSC framework)
        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
        confident_mask = (max_probs > args.pseudo_threshold)

        pseudo_loss = F.cross_entropy(pseudo_logits, pseudo_labels, reduction='none')
        pseudo_loss = (pseudo_loss * confident_mask).mean()

        if confident_mask.sum() <= 0:
            print(f"no pseudo labels were valuable enough to contribute, highest confidence was: {max_probs.max()}")
            global pseudo_miss  # TODO: prettify this because global variables arent really clean code
            pseudo_miss += 1

            # TODO: can I still do something for the pseudo_loss even if threshold is never met

        return pseudo_loss

    def pseudo_loss_S(self, y, pseudo_logits):
        # Same as pseudo_loss but uses S, which already is a distribution

        pseudo_loss = torch.tensor(0.0, device=device)
        if args.lmdb == 0.0:
            return pseudo_loss

        # s already provides some kind of distribution so just normalizing because it seemed to give errors
        pseudo_probs = pseudo_logits
        pseudo_probs = pseudo_probs / pseudo_logits.sum(dim=1, keepdim=True)

        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)

        topk_probs, topk_indices = torch.topk(pseudo_probs, k=args.topk, dim=1)
        confident_mask = (topk_probs.sum(dim=1) > args.pseudo_threshold
                          # * args.topk
                          )

        # Reweight top-k probs
        boosted_targets = pseudo_probs.clone()
        boosted_targets.scatter_(1, topk_indices, topk_probs * 2)  # Boost top-k samples
        boosted_targets = boosted_targets / boosted_targets.sum(dim=1, keepdim=True)

        loss_per_sample = F.kl_div(torch.log(pseudo_probs + 1e-8), boosted_targets.detach(),
                                   reduction='none').sum(dim=1)

        if confident_mask.any():
            pseudo_loss = (loss_per_sample * confident_mask).mean()
        else:
            print(f"[Pseudo] No samples met top-{args.topk} > {args.pseudo_threshold}. Max sum: {topk_probs.sum(dim=1).max():.4f}")
            global pseudo_miss
            pseudo_miss += 1

        # ece logging
        # if not hasattr(self, '_ece_call_count'):
        #     self._ece_call_count = 0
        # self._ece_call_count += 1
        #
        # if self._ece_call_count % 10 == 0:
        #     print(f"ECE vs true: {compute_ece(pseudo_probs.detach().cpu().numpy(), y)}")
        #     print(f"ECE vs argmax: {compute_ece(pseudo_probs.detach().cpu().numpy(), pseudo_labels.cpu())}")
        #
        #     acc = accuracy_score(torch.argmax(pseudo_probs.detach().cpu(), dim=1), y)
        #     print(f"[Pseudo] Mode: topk | Acc: {acc:.4f} | Loss: {pseudo_loss.item():.4f}")

        return pseudo_loss

    def cross_loss(self, y_pred, y):
        cross_loss = torch.tensor(0.0, device=device)
        if args.delta != 0.0:
            cross_loss = F.cross_entropy(y_pred,
                                         torch.from_numpy(y).long()) if not y_pred.numel() == 0 else torch.tensor(0)
        return cross_loss

    def supervised_contrastive_loss(self, s, labels, temperature=0.5):
        """
        Supervised contrastive loss on the soft assignment matrix `s`
        s: [N, C] soft cluster matrix
        labels: [N] ground-truth labels for supervised samples
        """
        s = F.normalize(s, dim=1)
        sim_matrix = torch.matmul(s, s.T) / temperature  # cosine similarity -> becomes O(n**2)

        # Exclude self-comparisons
        mask = torch.eye(len(labels), device=s.device).bool()
        # mixed precision, use -9e15 if you do not plan on using it
        sim_matrix.masked_fill_(mask, -1e4)

        # Create mask of positive pairs (same label)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        label_mask.fill_diagonal_(False)  # remove self

        # Compute log-softmax over similarity matrix
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Only keep log-probs of positives
        loss = - (log_prob * label_mask.float()).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
        return loss.mean()

    def fixmatch_style_loss(self, y, S, pseudo_logits, threshold=.95):

        fm_loss = torch.tensor(0.0, device=device)

        S_probs = S / S.sum(dim=1, keepdim=True)
        max_probs, pseudo_labels = torch.max(S_probs, dim=1)
        mask = max_probs >= threshold

        if mask.sum() == 0:
            print(f"[FM-style] No samples met {threshold} threshold. Max sum: {max_probs.max():.4f}")
            return fm_loss

        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        fm_loss = F.cross_entropy(pseudo_probs[mask], pseudo_labels[mask])

        return fm_loss


def spatial_filter(data_matrix, location, image_size):
    data_matrix = data_matrix.reshape([data_matrix.shape[0], -1])
    S_matrix = torch.zeros([image_size[0], image_size[1], data_matrix.shape[1]],device=device)
    Mask = torch.zeros([image_size[0], image_size[1]],device=device)
    Mask[location[:, 0], location[:, 1]] = 1
    S_matrix[location[:, 0], location[:, 1], :] = data_matrix
    import kornia as k
    S_matrix = S_matrix.permute(2, 0, 1)
    S_matrix = torch.unsqueeze(S_matrix, dim=0)
    matrix_blur = k.filters.box_blur(S_matrix, kernel_size=(args.smooth_window_size, args.smooth_window_size))
    matrix_blur = torch.squeeze(matrix_blur)
    matrix_blur = matrix_blur.permute(1, 2, 0)
    Mask = torch.unsqueeze(Mask, dim=0)
    Mask = torch.unsqueeze(Mask, dim=0)
    Mask = k.filters.box_blur(Mask, kernel_size=(args.smooth_window_size, args.smooth_window_size))
    Mask = torch.squeeze(Mask)
    Mask = torch.broadcast_to(Mask, [data_matrix.shape[1], Mask.shape[0], Mask.shape[1]])
    Mask = Mask.permute(1, 2, 0)
    Mask = matrix_blur / Mask
    new_target = Mask[location[:, 0], location[:, 1], :]
    torch.cuda.empty_cache()
    return new_target


def refined_subspace_affinity(s):
    weight = s ** 2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    train_loader = DataLoader(
        dataset, batch_size=4096, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(args.pre_train_iters):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.type(torch.float32)
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))

def train_one_epoch_fixmatch(model, unlabeled_data, scaler, batch_size=64, threshold=0.95, temperature=1.0, device='cuda:0'):
    """
    Apply FixMatch-style training for one epoch.

    Args:
        model (C_EDESC): The model to train.
        unlabeled_data (Tensor): Tensor of shape [N, C, H, W].
        threshold (float): Confidence threshold for pseudo-labeling.
        temperature (float): Softmax temperature scaling.
        device (str): Device to run computations on.

    Returns:
        fixmatch_loss (torch.Tensor): Average FixMatch loss for this epoch.
    """

    model.train()

    weak_aug = WeakAugGPU()
    strong_aug = HSIRandAugGPU()

    # Wrap tensor in a Dataset and create DataLoader
    unlabeled_dataset = TensorDataset(unlabeled_data)
    loader = DataLoader(unlabeled_dataset, batch_size=len(unlabeled_data), shuffle=True)

    total_loss = 0.
    total_count = 0

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)

        # Apply weak and strong augmentations
        weak_batch = weak_aug(x_batch)
        strong_batch = strong_aug(x_batch)

        # Forward weakly augmented batch for pseudo-labels
        _, s_weak, _, _ = model(weak_batch)
        probs = s_weak.detach() / s_weak.detach().sum(dim=1, keepdim=True)

        max_probs, pseudo_labels = torch.max(probs, dim=1)

        mask = max_probs >= threshold

        if mask.sum() == 0:
            print(f"[FM] No samples met {threshold} threshold. Max sum: {max_probs.max():.4f}")
            continue  # Skip this batch if no confident pseudo-labels

        # Forward strongly augmented inputs
        _, s_strong, _, _ = model(strong_batch[mask])

        loss = F.cross_entropy(s_strong, pseudo_labels[mask])
        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()

    # Normalize total loss
    if total_count > 0:
        return torch.tensor(total_loss / total_count)
    else:
        return torch.tensor(0.0, device=device)

def generate_cluster_pseudo_labels(
    data_all,        # np.ndarray, shape (num_samples, feat_dim)
    cluster_ids,     # array-like, shape (num_samples,)
    mask_lab,        # bool array, True for labeled
    true_labels,     # int array, ground truth for labeled (junk/–1 for unlabeled)
    N=5              # # of neighbors per seed
):
    num = data_all.shape[0]
    # full array to accumulate: -1 = no label, ≥0 = label
    full_labels = -1 * np.ones(num, dtype=int)
    full_labels[mask_lab] = true_labels[mask_lab]

    # for each cluster, propagate
    for c in np.unique(cluster_ids):
        idx_c      = np.where(cluster_ids == c)[0]
        lab_idx    = idx_c[mask_lab[idx_c]]
        unlab_idx  = idx_c[~mask_lab[idx_c]]
        if len(lab_idx)==0 or len(unlab_idx)==0:
            continue

        feats_lab   = data_all[lab_idx].reshape(len(lab_idx), -1)
        feats_unlab = data_all[unlab_idx].reshape(len(unlab_idx), -1)

        nn = NearestNeighbors(n_neighbors=min(N, len(unlab_idx)), algorithm='auto')
        nn.fit(feats_unlab.cpu())
        dists, nbrs = nn.kneighbors(feats_lab.cpu())

        for i, li in enumerate(lab_idx):
            neigh_glob = unlab_idx[nbrs[i]]
            # assign seed’s label
            full_labels[neigh_glob] = true_labels[li]

    # build outputs
    pseudo_mask_full = (full_labels >= 0) & (~mask_lab)        # unlabeled→now pseudo-labeled
    pseudo_indices   = np.where(pseudo_mask_full)[0]          # their positions
    pseudo_targets   = full_labels[pseudo_mask_full]          # the labels themselves

    return pseudo_targets, pseudo_mask_full, pseudo_indices
from sklearn.neighbors import NearestNeighbors
import numpy as np

def generate_cluster_pseudo_labels2(
    data_all,            # np.ndarray, shape (num_samples, feat_dim)
    c_all,               # np.ndarray, shape (num_samples, num_levels) from FINCH()
    mask_lab,            # bool array, True for labeled
    true_labels,         # int array, ground truth for labeled (-1 for unlabeled)
    mini_level=0         # which FINCH level to use for mini‐clusters
):
    """
    Propagate each labeled sample’s true label to its entire FINCH mini-cluster.

    Args:
        data_all      : features (unused here, but kept for API parity)
        c_all         : hierarchical FINCH cluster IDs (num_samples × num_levels)
        mask_lab      : boolean mask of which samples are labeled
        true_labels   : array of true labels (only valid where mask_lab True)
        mini_level    : which FINCH hierarchy level to treat as ‘mini‐cluster’

    Returns:
        pseudo_targets    : array of pseudo‐labels for formerly unlabeled
        pseudo_mask_full  : boolean mask of newly pseudo‐labeled samples
        pseudo_indices    : indices of those pseudo‐labeled samples
    """
    num = data_all.shape[0]
    # initialize everything as “no label”
    full_labels = -1 * np.ones(num, dtype=int)
    # set the ground‐truth labels
    full_labels[mask_lab] = true_labels[mask_lab]

    # extract the mini‐cluster IDs for every point
    mini_ids = c_all[:, mini_level]

    # for each labeled sample, assign its label to all other points
    # in its mini‐cluster
    labeled_indices = np.where(mask_lab)[0]
    for li in labeled_indices:
        this_label = true_labels[li]
        # members of the same mini‐cluster
        members = np.where(mini_ids == mini_ids[li])[0]
        # only keep the unlabeled ones
        unlab_members = members[~mask_lab[members]]
        full_labels[unlab_members] = this_label

    # build outputs
    pseudo_mask_full = (full_labels >= 0) & (~mask_lab)
    pseudo_indices   = np.where(pseudo_mask_full)[0]
    pseudo_targets   = full_labels[pseudo_mask_full]

    return pseudo_targets, pseudo_mask_full, pseudo_indices


def train_EDESC(device, i):

    start = time.time()

    # setting seed for the first round so only the first round is reproducible.
    # Any other round will use a random seed.
    if i == 0:
        seed = 42
        setup_seed(seed)
    else:
        seed = None

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(log_dir=f"tensorboard/{args.dataset}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

    model = C_EDESC(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain_path=args.pretrain_path).to(device)

    data = dataset.x
    data = data.astype(np.float16)
    y = dataset.y


    ascii_histogram(y)
    # utilize finch to generate big groups
    from finch import FINCH
    data = data.reshape(data.shape[0], -1)
    c, num_clust, req_c1 = FINCH(data)
    req_c = c[:, args.hierarchy]

    # model.pretrain('') # empty path will pretrain again
    model.pretrain(f'original_weight/{args.dataset}.pkl')

    optimizer = Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    scaler = GradScaler()

    index = dataset.index
    data = dataset.train.astype(np.float16)
    data = torch.Tensor(data).to(device)
    x_bar, hidden = get_initial_value(model, data)

    # TODO: chances are that not all classes are within the l_feats selection for percentage wise selection
    u_feats, l_feats, l_targets, mask_lab = get_labeled_data_strat(y, data, args.label_usage, random_state=seed)
    # print("mask;", np.where(mask_lab)[0])

    if args.pseudo_clusters == True:
        # expanding l_feats based on nearest neighbors in FINCH-clusters
        pseudo_targets, pseudo_mask_full, pseudo_indices = generate_cluster_pseudo_labels(
            data.reshape(len(data), -1),
            cluster_ids=req_c,
            mask_lab=mask_lab,
            true_labels=np.where(mask_lab, y, -1),
            # N=(args.label_usage if args.label_usage >= 1 else (data.y.shape[0] * args.label_usage) / args.n_clusters)
            N = args.pseudo_nn
        )
        pseudo_targets, pseudo_mask_full, pseudo_indices = generate_cluster_pseudo_labels2(
            data.reshape(dataset.__len__(), -1),
            c_all=c,
            mask_lab=mask_lab,
            true_labels=np.where(mask_lab, y, -1),
            mini_level = 1
        )

        print(accuracy_score(pseudo_targets, y[pseudo_indices]))

        # masking out the new pseudo_features
        u_idx = np.where(~mask_lab)[0]
        pseudo_mask_unlabeled = pseudo_mask_full[u_idx]
        pseudo_feats = u_feats[pseudo_mask_unlabeled]

        # adding the pseudo_feats and pseudo_targets to l_feats and l_targets
        print(l_feats.shape, u_feats.shape, pseudo_feats.shape)
        l_feats = torch.cat([l_feats, pseudo_feats], dim=0)
        l_targets = torch.cat([torch.tensor(l_targets), torch.tensor(pseudo_targets)], dim=0)

        # shrinking u_feats and expanding the mask_lab
        u_feats = u_feats[~pseudo_mask_unlabeled]
        mask_lab = mask_lab | pseudo_mask_full
        print(l_feats.shape, u_feats.shape)

        # print("labeled data distribution")
        # ascii_histogram(l_targets.numpy())

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=seed)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy().reshape(dataset.__len__(), -1))
    print("Initial Cluster Centers: ", y_pred)

    kmeans = SemiSupKMeans(k=args.n_clusters, tolerance=1e-4, max_iterations=300, init='k-means++',
                           n_init=30, random_state=seed, n_jobs=8, pairwise_batch_size=512,
                           mode=None)

    u_feats = torch.tensor(u_feats, dtype=torch.float32)
    l_feats = torch.tensor(l_feats, dtype=torch.float32)
    l_targets = torch.tensor(l_targets)

    kmeans.fit_mix(u_feats.view(u_feats.size(0), -1), l_feats.view(l_feats.size(0), -1), l_targets)
    y_pred_sskm = kmeans.labels_.cpu().numpy()

    print(f"kmeans: {accuracy_score(y_pred, y)}")
    print(f"sskmeans both: {accuracy_score(y_pred_sskm, y)}")

    global km_acc, km_nmi, km_kappa
    km_acc.append(accuracy_score(y_pred, y))
    km_nmi.append(normalized_mutual_info_score(y_pred, y))
    km_kappa.append(cohen_kappa_score(y_pred, y))

    global semi_acc, semi_nmi, semi_kappa
    semi_acc.append(accuracy_score(y_pred_sskm, y))
    semi_nmi.append(normalized_mutual_info_score(y_pred_sskm,y))
    semi_kappa.append(cohen_kappa_score(y_pred_sskm, y))

    # use ss-k-means over unsupervised version
    del kmeans
    if args.semsc:
        y_pred = y_pred_sskm

    # Initialize D
    D = Initialization_D(hidden.reshape(len(y_pred), -1), y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)

    accmax = 0
    nmimax = 0
    kappa_max = 0
    ca_max = []
    torch.cuda.empty_cache()
    model.D.data = D.to(device)
    model.train()
    from torch_scatter import scatter_mean
    max_ratio = 0

    for epoch_iter in range(epochs):

        with autocast():
            x_bar, (s, s_weak), z, (pseudo_logits, pseudo_logits_aug) = model(data)

            ratio = (s > 0.90).sum() / s.shape[0]

            # fancy indexing get the mean prediction of a mini-cluster
            original_center = scatter_mean(src=s, index=torch.tensor(req_c, dtype=torch.int64, device=device), dim=0)
            refined_center = refined_subspace_affinity(original_center)

            y_pred = s.cpu().detach().numpy().argmax(1)
            y_pred_remap, acc, kappa, nmi, ca, mapping = cluster_accuracy(y, y_pred, return_aligned=True)

            # originally this was just ration> max_ratio, though increasing accuracy should also be important(?)
            if ratio > max_ratio and acc > accmax:
                accmax = acc
                kappa_max = kappa
                nmimax = nmi
                ca_max = ca
                max_ratio = ratio

            # for cross_entropy we need the logits of y_pred_remap so applying the mapping dict
            logits = s.cpu()

            # Create permutation index array to remap the logits
            num_classes = logits.size(1)
            perm = [0] * num_classes
            for old_index, new_index in mapping.items():
                perm[new_index] = old_index

            # Convert to tensor and ensure it's on the same device as logits
            perm = torch.tensor(perm, dtype=torch.long, device=logits.device)
            # Apply permutation to logits
            y_pred_logits_remap = logits[:, perm]

            # Train test splitting as an arbitrary way of limiting data
            y_pred_logits_remap_partial = y_pred_logits_remap[mask_lab]
            y_partial = y[mask_lab] #l_targets is a tensor, while this requires a ndarray

            total_loss, reconstr_loss, kl_loss, loss_d1, loss_d2 = model.total_loss(
                x=data,
                x_bar=x_bar,
                center=original_center,
                target=refined_center,
                dim=args.d,
                n_clusters=args.n_clusters,
                s=s,
                index=index
            )

            cross_loss = torch.tensor(0.0)
            cross_loss = model.cross_loss(y_pred_logits_remap_partial,y_partial)

            # dont forget to apply the permutation here because the latent dim. and actual labels dont necessarily match
            # pseudo_logits[:, perm]

            pseudo_loss = torch.tensor(0.0)
            pseudo_loss = model.pseudo_loss_S(y,s[:, perm])

            contr_loss = torch.tensor(0.0)
            s_labeled = s[mask_lab]
            y_labeled = torch.tensor(l_targets, device=s.device)
            #maybe also use/add high conf pseudo labels?
            contr_loss = model.supervised_contrastive_loss(s_labeled[:,perm], y_labeled, temperature = .5)

            fixmatch_loss = torch.tensor(0.0)
            # augmentations on the soft-assignment matrix or latent dimension did not seem to help
            # fixmatch_loss =  model.fixmatch_style_loss(y, s_weak[:,perm], pseudo_logits_aug[:,perm], .80)
            fixmatch_loss =  model.fixmatch_style_loss(y, s[:,perm], pseudo_logits[:,perm], .80)


            # Combine the four semi-supervised losses with learned uncertainty weighting (not used)
            # semisup_losses = [cross_loss, pseudo_loss, contr_loss, fixmatch_loss]
            # weighted_semisup_loss = torch.tensor(0.0, device=device)
            #
            # for i, L in enumerate(semisup_losses):
            #     precision = torch.exp(-model.log_vars_semisup[i])  # = 1 / sigma_i^2
            #     # Weighted Loss = precision * L + log_vars_semisup[i] for regularization
            #     weighted_semisup_loss += precision * L + model.log_vars_semisup[i]
            # total_loss = (total_loss  + weighted_semisup_loss)

            total_loss = (total_loss +
                          args.delta * cross_loss +
                          args.lmdb * pseudo_loss +
                          args.psi * fixmatch_loss +
                          args.omega * contr_loss
                          )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # writer.add_scalar('Loss/total', total_loss.item(), epoch_iter)
            # writer.add_scalar('Loss/reconstruction', reconstr_loss.item(), epoch_iter)
            # writer.add_scalar('Loss/kl', kl_loss.item() * args.alpha, epoch_iter)
            # writer.add_scalar('Loss/CE', cross_loss.item() * args.delta, epoch_iter)
            # writer.add_scalar('Loss/pseudo', pseudo_loss.item() * args.lmdb, epoch_iter)
            # writer.add_scalar('Loss/contrastive', contr_loss.item() * args.omega, epoch_iter)
            # writer.add_scalar('Loss/FM', fixmatch_loss.item() * args.psi, epoch_iter)

            if epoch_iter % 10 == 0 or epoch_iter == epochs - 1:
                print('[Eval]', 'Iter {}'.format(epoch_iter), ':Current Acc {:.4f}'.format(acc),
                      ':Max Acc {:.4f}'.format(accmax), ', Current nmi {:.4f}'.format(nmi),
                      ':Max nmi {:.4f}'.format(nmimax), ', Current kappa {:.4f}'.format(kappa),
                      ':Max kappa {:.4f}'.format(kappa_max))
                print(
                    f"[Losses] | "
                    f"total: {total_loss.data:.4f} | "
                    f"recon: {reconstr_loss.data:.4f} | "
                    f"kl: {kl_loss.data:.4f} | "
                    f"entropy: {(args.delta * cross_loss.data):.4f} | "
                    f"pseudo: {(args.lmdb * pseudo_loss.data):.4f} | "
                    f"contras: {(args.omega * contr_loss.data):.4f} | "
                    f"FM: {(args.psi * fixmatch_loss.data):.4f}"
                )

                print("[Eval]", "ratio", ratio.data)

    end = time.time()

    # writer.close()
    print('Running time: ', end - start)

    # uncertainty weighting values
    print("weights: ",model.log_vars_semisup)

    global predicted_labels
    predicted_labels.append(y_pred_remap)

    cm = confusion_matrix(y, y_pred_remap)
    labels = np.unique(np.concatenate([y, y_pred_remap]))

    # calculating column width for confusion matrix formatting
    max_label_len = max(len(str(l)) for l in labels)
    max_count_len = max(len(str(v)) for v in cm.flatten())
    column_width = max(6, max_label_len + 2, max_count_len + 2)

    print("Confusion Matrix:")
    # header row: pad the empty corner to column_width, then each label
    print(" " * column_width + "".join(f"{l:>{column_width}}" for l in labels))
    for true_label, row in zip(labels, cm):
        # print true_label, then each count with the same width
        print(f"{true_label:>{column_width}}" + "".join(f"{v:>{column_width}d}" for v in row))

    if args.delta != .0 or args.omega != .0:
        print(f"{'percentage' if args.label_usage < 1 else 'number'} of labeled data used: {args.label_usage} meaning {len(y_pred_logits_remap_partial)}/{len(y)} used")

    del u_feats, l_feats, l_targets
    del c, num_clust, req_c
    return accmax, nmimax, kappa_max, ca_max


if __name__ == "__main__":
    import datetime
    torch.autograd.set_detect_anomaly(True)

    # Making sure seed has been set
    setup_seed(42)
    # Get current time
    now = datetime.datetime.now()
    print("hello world: " + str(now))

    pseudo_miss = 0

    semi_acc = []
    km_acc = []
    semi_nmi = []
    km_nmi = []
    semi_kappa = []
    km_kappa = []

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lnp', type=int, default=20)
    parser.add_argument('--outp', type=int, default=500)
    parser.add_argument('--smooth_window_size', type=int, default=7)
    parser.add_argument('--pre_train_iters', type=int, default=50)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--hierarchy', default=1, type=int)
    parser.add_argument('--n_z', default=32, type=int)
    parser.add_argument('--eta', default=5, type=int)
    parser.add_argument('--dataset', type=str, default='pavia')
    parser.add_argument('--device_index', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default='./tmp/flood/pre.pkl')
    parser.add_argument('--alpha', default=3, type=float, help='the weight of kl_loss')
    parser.add_argument('--beta', default=8, type=float, help='the weight of local_loss')
    parser.add_argument('--gama', default=0.03, type=float, help='the weight of non_local_loss')
    
    # semi-supervised parameters
    parser.add_argument('--label_usage', default=4, type=float, help='decimal% or absolute value of labeled data used during training')
    parser.add_argument('--semsc', default=True, type=bool, help='semi-supervised subspace construction')

    parser.add_argument('--delta', default=.56, type=float, help='the weight of the cross_entropy_loss')
    parser.add_argument('--omega', default=.09, type=float, help='the weight of the contrastive_loss')


    parser.add_argument('--pseudo_threshold', default=.95, type=float, help='minimum confidence of the pseudo predications to be used')
    parser.add_argument('--topk', default=1, type=int, help="count of pseudo labels to be summed for the threshold to be met")

    parser.add_argument('--lmdb', default=2.409, type=float, help='the weight of the pseudo_label_loss')
    parser.add_argument('--psi', default=0.98, type=float, help='the weight of the fixmatch-style_loss')

    #experimental
    parser.add_argument('--pseudo_clusters', default=False, type=bool, help='usage of pseudo-labels by FINCH')
    parser.add_argument('--pseudo_nn', default=2, type=int, help='Amount of neighbours in cluster to include in pseudolabel')

    #paramsearch
    parser.add_argument('--optuna', action='store_true', help='If set, run Optuna hyperparameter search instead of fixed args')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    if args.cuda:
        if args.device_index==0:
            device = torch.device("cuda:0")
        elif args.device_index==1:
            device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    print("current alpha: ", args.alpha, "current beta: ", args.beta, "current gama: ", args.gama)

    if args.dataset == 'Houston':
        args.pretrain_path = 'weight/Houston.pkl'
        args.n_clusters = 7
        args.n_input = 8
        args.image_size = [130, 130]
        dataset = Load_my_Dataset("C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/Houston/Houston_corrected.mat",
                                  "C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/Houston/Houston_gt.mat")
        args.num_sample = dataset.__len__()
    elif args.dataset == 'trento':
        # 0.001
        args.pretrain_path = 'weight/trento.pkl'
        args.n_clusters = 6
        args.n_input = 8
        args.image_size = [166, 600]
        dataset = Load_my_Dataset("C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/trento/Trento.mat",
                                  "C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/trento/Trento_gt.mat")
        args.num_sample = dataset.__len__()
    elif args.dataset == 'pavia':
        # 0.001
        args.pretrain_path = 'weight/pavia.pkl'
        args.n_clusters = 9
        args.n_input = 8
        args.image_size = [610, 340]
        dataset = Load_my_Dataset("C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/pavia/PaviaU.mat",
                                  "C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/pavia/PaviaU_gt.mat")
        args.num_sample = dataset.__len__()

    print(args)

    if args.optuna:
        # ── 1) Single‐run search ──
        def objective(trial):
            # sample hyperparams
            args.delta = trial.suggest_float("delta", 0.0, 1.0)
            args.lmdb = trial.suggest_float("lmdb", 0.1, 5.0, log=True)
            args.psi = trial.suggest_float("psi", 0.0, 1.0)
            args.omega = trial.suggest_float("omega", 0.0, 1.0)

            # one run per trial for speed
            seed = 42
            setup_seed(seed)
            acc, _, _, _ = train_EDESC(device=device, i=0)
            torch.cuda.empty_cache()
            return float(acc)

        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25, n_jobs=1)

        print("== Best hyperparameters (single-run estimate) ==")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")
        print(f"Estimated ACC: {study.best_value:.4f}\n")

        # ── 2) Final 5-seed re-evaluation of top 5 trials ──
        print("Re-evaluating top 3 candidates with 5 seeds each...")
        completed = [t for t in study.trials if t.value is not None]
        top3 = sorted(completed, key=lambda t: t.value, reverse=True)[:3]

        for rank, trial in enumerate(top3, start=1):
            params = trial.params
            accs = []
            nmis = []
            kappas = []
            for run in range(5):
                # setup_seed(100 + run)
                # apply candidate params
                args.delta, args.lmdb, args.psi, args.omega = (
                    params["delta"],
                    params["lmdb"],
                    params["psi"],
                    params["omega"]
                )

                acc, nmi, kappa, ca = train_EDESC(device=device, i=run)
                accs.append(acc)
                nmis.append(nmi)
                kappas.append(kappa)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Candidate #{rank}: {params} -> ACC = {mean_acc:.4f} +_ {std_acc:.4f}, NMI = {np.mean(nmis):.4f} +- {np.std(nmis):.4f}, KAPPA = {np.mean(kappas):.4f} +- {np.std(kappas):.4f}")
        end = time.time()
        print("time: ", end - start)
    else:

        bestacc = 0
        bestnmi = 0
        best_kappa = 0
        acc_sum = 0
        nmi_sum = 0
        kappa_sum = 0
        rounds = 5 # if stepping away from 1 round, unset the seeds every run

        cas = []
        accs = []
        nmis = []
        kappas = []

        predicted_labels = []

        for i in range(rounds):
            print("this is " + str(i) + "round")
            acc, nmi, kappa, ca = train_EDESC(device=device, i=i)
            accs.append(acc)
            nmis.append(nmi)
            kappas.append(kappa)
            cas.append(ca)

            if acc > bestacc:
                bestacc = acc
            if nmi > bestnmi:
                bestnmi = nmi
            if kappa > best_kappa:
                best_kappa = kappa

        cas = np.array(cas)
        ca = np.mean(cas, axis=0)
        print("cav:", ca)
        print("average_acc:", np.mean(accs), np.std(accs))
        print("average_nmi:", np.mean(nmis), np.std(nmis))
        print("average_kappa", np.mean(kappas), np.std(kappas))
        print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi), ' Best kappa {:4f}'.format(kappa))

        print(f"kmeans; avg_acc: {np.mean(km_acc):.4f}, avg_nmi: {np.mean(km_nmi):.4f}, avg_kappa: {np.mean(km_kappa):.4f}")
        print(f"ss-kmeans; avg_acc: {np.mean(semi_acc):.4f}, avg_nmi: {np.mean(semi_nmi):.4f}, avg_kappa: {np.mean(semi_kappa):.4f}")
        print(f'Pseudo classifier missed threshold {args.pseudo_threshold} {pseudo_miss/rounds if args.lmdb != .0 else 0} out of {epochs if args.lmdb != .0 else 0} times')

        from scipy.stats import mode
        y_pred = mode(predicted_labels, axis=0, keepdims=False).mode
        print(y_pred.shape)

        img = np.full((args.image_size[0], args.image_size[1]), 0, dtype=dataset.y.dtype)
        rows, cols = dataset.index
        img[rows, cols] = y_pred+1

        import scipy.io
        # scipy.io.savemat(f'./results/conference/01-{args.dataset}-SCDSC.mat', {'prediction': img})