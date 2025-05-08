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

from attention_span_classifier import AttentionPseudoClassifier

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

from faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from cluster_and_log_utils import log_accs_from_preds
from sklearn.metrics import accuracy_score
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

        # augmentation functions called in foward pass, they did not really seem to help...
        self.weak_latent_aug = WeakLatentAug(noise_std=0.01, flip_prob=0.5)
        self.strong_latent_aug = StrongLatentAug(
            noise_std=0.1,
            drop_prob=0.2,

            # TODO:these probably do something, but i dont feel like its noticable.. even on paviaU
            #  maybe more labeled data/finetuning the loss?
            shift=True,
            flip=True
        )

        # learnable weights of the loss func
        # cross, pseudo, contrastive, fixmatch
        # self.log_vars_semisup = nn.Parameter(torch.zeros(4))
        self.log_vars_semisup = nn.Parameter(torch.tensor([args.delta, args.epsilon, args.omega, args.psi]))

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
        # self.pseudo_classifier = AttentionPseudoClassifier(n_z, n_clusters)

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
        # z_aug = z + torch.randn_like(z) * 0.1
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
        if args.epsilon == .0:
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
        if args.epsilon == 0.0:
            return pseudo_loss

        # s already provides some kind of distribution so just normalizing
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

def train_one_epoch_fixmatch_minibatched(model, unlabeled_data, scaler, batch_size=64, threshold=0.95, temperature=1.0, device='cuda:0'):
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

    # stopped developing this function, depreciated

    model.train()

    # weak_aug = WeakAug()
    # strong_aug = HSIRandAugment()

    weak_aug = WeakAugGPU()
    strong_aug = HSIRandAugGPU()

    # Wrap tensor in a Dataset and create DataLoader
    unlabeled_dataset = TensorDataset(unlabeled_data)
    loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.
    total_count = 0

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)

        # Apply weak and strong augmentations
        # weak_batch = torch.stack([weak_aug(x.clone().to(device)) for x in x_batch])
        # strong_batch = torch.stack([strong_aug(x.clone().to(device)) for x in x_batch])
        weak_batch = weak_aug(x_batch)
        strong_batch = strong_aug(x_batch)

        # Forward weakly augmented batch for pseudo-labels
        with autocast():
            _, s_weak, _, _ = model(weak_batch)
            probs = F.softmax(s_weak / temperature, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)

            mask = max_probs >= threshold

            if mask.sum() == 0:
                continue  # Skip this batch if no confident pseudo-labels

            # Forward strongly augmented inputs
            _, s_strong, _, _ = model(strong_batch[mask])

            loss = F.cross_entropy(s_strong, pseudo_labels[mask])
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()

        # Backprop
        scaler.scale(loss).backward()
        scaler.step(model.optimizer)
        scaler.update()
        model.optimizer.zero_grad()

    # Normalize total loss
    if total_count > 0:
        print(f"[FM] No samples met {threshold} threshold. Max sum: {max_probs.max():.4f}")
        return torch.tensor(total_loss / total_count)
    else:
        print(f"skipped fixmatch")
        return torch.tensor(0.0, device=device)
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
        # weak_batch = torch.stack([weak_aug(x.clone().to(device)) for x in x_batch])
        # strong_batch = torch.stack([strong_aug(x.clone().to(device)) for x in x_batch])
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


def train_EDESC(device, i):

    start = time.time()

    # for whatever reason this function does not take global scope, so this is required for reproducibility
    seed = 42
    setup_seed(seed)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="tensorboard")

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

    model.pretrain(f'original_weight/{args.dataset}.pkl')
    # model.pretrain('') # empty path will pretrain again

    optimizer = Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    # optimizer_classifier = Adam(model.pseudo_classifier.parameters(), lr=args.lr)
    # optimizer_clustering = Adam([model.D], lr=args.lr)

    scaler = GradScaler()

    index = dataset.index
    data = dataset.train
    data = torch.Tensor(data).to(device)
    x_bar, hidden = get_initial_value(model, data)

    print(f'seed: {np.random.get_state()[1][0]}')
    # TODO: chances are that not all classes are within the l_feats selection for percentage wise selection
    u_feats, l_feats, l_targets, mask_lab = get_labeled_data_strat(y, data, args.label_usage, seed)
    print("mask;", np.where(mask_lab)[0])

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=seed)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy().reshape(dataset.__len__(), -1))
    print("Initial Cluster Centers: ", y_pred)

    kmeans = SemiSupKMeans(k=args.n_clusters, tolerance=1e-4, max_iterations=300, init='k-means++',
                           n_init=30, random_state=seed, n_jobs=-1, pairwise_batch_size=1024,
                           mode=None)

    u_feats = torch.tensor(u_feats, dtype=torch.float32)
    l_feats = torch.tensor(l_feats, dtype=torch.float32)
    l_targets = torch.tensor(l_targets)

    kmeans.fit_mix(u_feats.view(u_feats.size(0), -1), l_feats.view(l_feats.size(0), -1), l_targets)
    all_preds = kmeans.labels_.cpu().numpy()

    print(f"kmeans: {accuracy_score(y_pred, y)}")
    print(f"sskmeans both: {accuracy_score(all_preds, y)}")

    # TODO: ss-k-means does not increase performance at all
    #  further more it is also inconsistent in being better for more labeled data...
    #  maybe if labeled data is an absolute value per class
    y_pred = all_preds

    # Initialize D
    D = Initialization_D(hidden.reshape(dataset.__len__(), -1), y_pred, args.n_clusters, args.d)
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
            logits = s.cpu()#.detach().numpy() #is detaching allowed/necessary?

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

            cross_loss = model.cross_loss(y_pred_logits_remap_partial,y_partial)

            # # dont forget to apply the permutation here because the latent dim. and actual labels dont necessarily match
            # # pseudo_logits[:, perm]
            #
            pseudo_loss = torch.tensor(0.0)
            fixmatch_loss = torch.tensor(0.0)
            pseudo_loss = model.pseudo_loss_S(
                y,
                s[:, perm]
            )

            #
            s_labeled = s[mask_lab]
            y_labeled = torch.tensor(l_targets, device=s.device)
            contr_loss = model.supervised_contrastive_loss(s_labeled, y_labeled) #maybe also use/add high conf pseudo labels?
            #
            fixmatch_loss =  model.fixmatch_style_loss(y, s[:,perm], pseudo_logits[:,perm], .80)
            # # fixmatch_loss =  model.fixmatch_style_loss(y, s_weak[:,perm], pseudo_logits_aug[:,perm], .80)


            # Combine the four semi-supervised losses with learned uncertainty weighting
            # semisup_losses = [cross_loss, pseudo_loss, contr_loss, fixmatch_loss]
            # weighted_semisup_loss = torch.tensor(0.0, device=device)

            # for i, L in enumerate(semisup_losses):
            #     precision = torch.exp(-model.log_vars_semisup[i])  # = 1 / sigma_i^2
            #     # Weighted Loss = precision * L + log_vars_semisup[i] for regularization
            #     weighted_semisup_loss += precision * L + model.log_vars_semisup[i]

            # Your normal main losses:
            # total_loss = (total_loss  # e.g. reconstruction + alpha*KL + beta*smooth
            #               + weighted_semisup_loss
            #               )

            total_loss  = (total_loss +
                           args.delta * cross_loss +
                           args.epsilon * pseudo_loss +
                           args.psi * fixmatch_loss +
                           args.omega * contr_loss
                           )

            # als multi optimizerl gebruiken dan zero loss kopelen met s.sum()*.0 zodanig dat graph verbonden
            # blijft maar die run is dan "useless", natuurlijk ok want pseudo_label ging niet boven thresh
            # dus er ging sws niet gebeuren

            scaler.scale(total_loss).backward()
            # scaler.scale(total_loss).backward(retain_graph=True)
            # scaler.scale(fixmatch_loss).backward()
            # scaler.scale(pseudo_loss).backward()

            scaler.step(optimizer)
            # if any(p.grad is not None for p in model.pseudo_classifier.parameters()):
            #     scaler.step(optimizer_classifier)
            # if any(p.grad is not None for p in [model.D]):
            #     scaler.step(optimizer_clustering)

            scaler.update()
            optimizer.zero_grad()
            # optimizer_classifier.zero_grad()
            # optimizer_clustering.zero_grad()

            # writer.add_scalar('Loss/total', total_loss.item(), epoch_iter)
            # writer.add_scalar('Loss/reconstruction', reconstr_loss.item(), epoch_iter)
            # writer.add_scalar('Loss/kl', kl_loss.item() * args.alpha, epoch_iter)
            # writer.add_scalar('Loss/CE', cross_loss.item() * args.delta, epoch_iter)
            # writer.add_scalar('Loss/pseudo', pseudo_loss.item() * args.epsilon, epoch_iter)
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
                    f"pseudo: {(args.epsilon * pseudo_loss.data):.4f} | "
                    f"contras: {(args.omega * contr_loss.data):.4f} | "
                    f"FM: {(args.psi * fixmatch_loss.data):.4f}"
                )

                # print(
                #     f"[Losses] | "
                #     f"total: {total_loss.data:.4f} | "
                #     f"recon: {reconstr_loss.data:.4f} | "
                #     f"kl: {kl_loss.data:.4f} | "
                #     f"entropy: {(cross_loss.data):.4f} | "
                #     f"pseudo: {(pseudo_loss.data):.4f} | "
                #     f"contras: {(contr_loss.data):.4f} | "
                #     f"FM: {(fixmatch_loss.data):.4f}"
                # )
                print("[Eval]", "ratio", ratio.data)

    end = time.time()

    writer.close()
    print('Running time: ', end - start)

    # uncertainty weighting weights
    # print(model.log_vars_semisup)


    cm = confusion_matrix(y, y_pred_remap)
    labels = np.unique(np.concatenate([y, y_pred_remap]))
    print("Confusion Matrix:")
    # header row
    print("\t" + "\t".join(str(l) for l in labels))
    # each row: true label, then counts per predicted label
    for true_label, row in zip(labels, cm):
        print(f"{true_label}\t" + "\t".join(str(v) for v in row))

    if args.delta != .0 or args.omega != .0:
        print(f"{'percentage' if args.label_usage < 1 else 'number'} of labeled data used: {args.label_usage} meaning {len(y_pred_logits_remap_partial)}/{len(y)} used")

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
    parser.add_argument('--dataset', type=str, default='Houston')
    parser.add_argument('--device_index', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default='./tmp/flood/pre.pkl')
    parser.add_argument('--alpha', default=3, type=float, help='the weight of kl_loss')
    parser.add_argument('--beta', default=8, type=float, help='the weight of local_loss')
    parser.add_argument('--gama', default=0.03, type=float, help='the weight of non_local_loss')
    # semi-supervised parameters
    parser.add_argument('--label_usage', default=4, type=float, help='decimal% or absolute value of labeled data used during training')
    parser.add_argument('--delta', default=.5, type=float, help='the weight of the cross_entropy_loss')
    parser.add_argument('--epsilon', default=2 , type=float, help='the weight of the pseudo_label_loss') #2
    parser.add_argument('--pseudo_threshold', default=.95, type=float, help='minimum confidence of the pseudo predications to be used') #.95
    parser.add_argument('--topk', default=1, type=int, help="count of pseudo labels to be summed for the threshold to be met") #1
    parser.add_argument('--psi', default=.4, type=float, help='the weight of the fixmatch-style_loss') #.5
    parser.add_argument('--omega', default=.3, type=float, help='the weight of the contrastive_loss') #.5

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
    elif args.dataset == 'indian':
        args.pretrain_path = 'original_weight/indian_pines.pkl'
        args.n_clusters = 16
        args.n_input = 8
        args.image_size = [145, 145]
        dataset = Load_my_Dataset("C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/extra/indian_pines/Indian_pines.mat",
                                  "C:/Users/thoma/Documents/School/Master/MasterProef/codebase/HSI/extra/indian_pines/Indian_pines_gt.mat",)

    print(args)
    bestacc = 0
    bestnmi = 0
    best_kappa = 0
    acc_sum = 0
    nmi_sum = 0
    kappa_sum = 0
    rounds = 1 # if stepping away from 1 round, unset the seeds every run
    cas = []
    for i in range(rounds):
        print("this is " + str(i) + "round")
        acc, nmi, kappa, ca = train_EDESC(device=device, i=i)
        acc_sum = acc_sum + acc
        nmi_sum = nmi_sum + nmi
        cas.append(ca)
        kappa_sum = kappa_sum + kappa
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
        if kappa > best_kappa:
            best_kappa = kappa
    cas = np.array(cas)
    ca = np.mean(cas, axis=0)
    print("cav:", ca)
    average_acc = acc_sum / rounds
    average_nmi = nmi_sum / rounds
    average_kappa = kappa_sum / rounds
    print("average_acc:", average_acc)
    print("average_nmi:", average_nmi)
    print("average_kappa", average_kappa)
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi), ' Best kappa {:4f}'.format(kappa))
    print(f'Pseudo classifier missed threshold {args.pseudo_threshold} {pseudo_miss if args.epsilon != .0 else 0} out of {epochs if args.epsilon != .0 else 0} times')