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
from torch.utils.data import DataLoader

from attention_span_classifier import AttentionPseudoClassifier
from my_knn import get_initial_value
from Auto_encoder import Ae
from getdata import Load_my_Dataset
from utils import cluster_accuracy, ascii_histogram
import warnings
from Initialize_D import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
import os
import random

from faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from cluster_and_log_utils import log_accs_from_preds
from sklearn.metrics import accuracy_score


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

        # TODO: Pseudo-Graph Module, requires mini-batch integration
        # Makes it O(n**2) again, so best to avoid? Though FINCH is 0(n**2) as wel
        # proposed method uses small set of samples to train network
        # Not end to end trainable anymore
        # self.Coef = Parameter(torch.Tensor(batch_size,batch_size))
        # nn.init.constant_(self.Coef, 1e-5)

        # Pseudo-Label Module
        latent_dim = n_z * 7 * 7
        # self.pseudo_classifier = nn.Linear(latent_dim, n_clusters)
        # Not used: using S instead
        # self.pseudo_classifier = nn.Sequential(
        #     nn.Conv2d(n_z, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.3),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(32, n_clusters)
        # )
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

        # Pseudo Label
        # pseudo_logits = self.pseudo_classifier(
        #     z#.detach()
        # )

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

        return x_bar, s, z, 0

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

    def pseudo_loss_adv(self, y, pseudo_logits):

        pseudo_loss = torch.tensor(0.0, device=device)
        if args.epsilon == .0:
            return pseudo_loss


        # print(f"Pseudo accuracy was {accuracy_score(torch.argmax(pseudo_logits.cpu(), dim=1), y)}")

        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)

        print(f"ece score is: {compute_ece(pseudo_probs.detach().cpu().numpy(), y)}")
        print(f"ece score is: {compute_ece(pseudo_probs.detach().cpu().numpy(), pseudo_labels.cpu())}")

        # Option 1: Soft labeling with confidence weighting
        if args.pseudo_mode == 'soft_weighted':
            # Each sample contributes weighted by its confidence
            targets = pseudo_probs.detach()  # detach to avoid gradient on targets
            weights = max_probs.detach()
            loss = F.kl_div(F.log_softmax(pseudo_logits, dim=1), targets, reduction='none').sum(dim=1)
            pseudo_loss = (loss * weights).mean()

        # Option 2: Require top-k probabilities to sum to threshold
        elif args.pseudo_mode == 'topk_sum':
            # DEFAULT
            # topk_probs, _ = torch.topk(pseudo_probs, k=args.topk, dim=1)
            # confident_mask = (topk_probs.sum(dim=1) > args.pseudo_threshold)
            #
            # targets = pseudo_probs.detach()
            # loss = F.kl_div(F.log_softmax(pseudo_logits, dim=1), targets, reduction='none').sum(dim=1)
            #
            # if confident_mask.any():
            #     pseudo_loss = (loss * confident_mask).mean()
            # else:
            #     print(f"[Pseudo] No samples met top-{args.topk} threshold. Max sum: {topk_probs.sum(dim=1).max():.4f}")
            #     global pseudo_miss
            #     pseudo_miss += 1

            # MASKING
            # topk_probs, topk_indices = torch.topk(pseudo_probs, k=args.topk, dim=1)
            # confident_mask = topk_probs.sum(dim=1) > args.pseudo_threshold
            #
            # # Create masked + renormalized targets
            # masked_targets = torch.zeros_like(pseudo_probs)
            # masked_targets.scatter_(1, topk_indices, topk_probs)  # keep top-k values
            # masked_targets = masked_targets / masked_targets.sum(dim=1, keepdim=True)  # normalize
            #
            # # Compute KL loss (only for confident samples)
            # student_log_probs = F.log_softmax(pseudo_logits, dim=1)
            # loss_per_sample = F.kl_div(student_log_probs, masked_targets.detach(), reduction='none').sum(dim=1)
            #
            # if confident_mask.any():
            #     pseudo_loss = (loss_per_sample * confident_mask).mean()
            # else:
            #     pseudo_loss = torch.tensor(0.0, device=pseudo_logits.device)
            #     print(f"[Pseudo] No samples met top-{args.topk} threshold. Max sum: {topk_probs.sum(dim=1).max():.4f}")
            #     global pseudo_miss
            #     pseudo_miss += 1

            # BOOSTING
            topk_probs, topk_indices = torch.topk(pseudo_probs, k=args.topk, dim=1)
            confident_mask = topk_probs.sum(dim=1) > args.pseudo_threshold

            # Boost top-k probs
            boost_factor = 2.0  # You can tune this
            boosted_targets = pseudo_probs.clone()
            boosted_targets.scatter_(
                1,
                topk_indices,
                pseudo_probs.gather(1, topk_indices) * boost_factor
            )
            boosted_targets = boosted_targets / boosted_targets.sum(dim=1, keepdim=True)

            # Compute KL loss
            student_log_probs = F.log_softmax(pseudo_logits, dim=1)
            loss_per_sample = F.kl_div(student_log_probs, boosted_targets.detach(), reduction='none').sum(dim=1)

            if confident_mask.any():
                pseudo_loss = (loss_per_sample * confident_mask).mean()
            else:
                pseudo_loss = torch.tensor(0.0, device=pseudo_logits.device)
                print(f"[Pseudo] No samples met top-{args.topk} threshold. Max sum: {topk_probs.sum(dim=1).max():.4f}")
                global pseudo_miss
                pseudo_miss += 1

        else:
            raise ValueError(f"Unknown pseudo_mode: {args.pseudo_mode}")

        acc = accuracy_score(torch.argmax(pseudo_logits.detach().cpu(), dim=1), y)
        print(f"[Pseudo] Mode: {args.pseudo_mode} | Pseudo acc: {acc:.4f} | Loss: {pseudo_loss.item():.4f}")

        return pseudo_loss

    def pseudo_loss_adv2(self, y, pseudo_logits):
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
            print(f"[Pseudo] No samples met top-{args.topk} threshold. Max sum: {topk_probs.sum(dim=1).max():.4f}")
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

    def contrastive_loss_topk(self, s, temperature=0.5, k=3):
        """
        Contrastive loss using top-k most similar as positives
        s: [batch_size, num_clusters] soft cluster assignment matrix
        """
        s = F.normalize(s, dim=1)  # [B, C]
        sim_matrix = torch.matmul(s, s.T) / temperature  # [B, B]

        batch_size = s.size(0)
        mask = torch.eye(batch_size, device=s.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)  # remove self-similarity

        # Get top-k positive indices
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)

        # Create target distribution for positives
        pos_mask = torch.zeros_like(sim_matrix)
        pos_mask.scatter_(1, topk_indices, 1.0)

        # Apply log-softmax to similarity
        logits = F.log_softmax(sim_matrix, dim=1)

        # Compute contrastive loss as the average log-prob of positives
        loss = - (logits * pos_mask).sum(dim=1) / k
        return loss.mean()

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
        sim_matrix.masked_fill_(mask, -9e15)

        # Create mask of positive pairs (same label)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        label_mask.fill_diagonal_(False)  # remove self

        # Compute log-softmax over similarity matrix
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Only keep log-probs of positives
        loss = - (log_prob * label_mask.float()).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
        return loss.mean()


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

def train_one_epoch_fixmatch(X, train_size):
    # TODO: - future work - from consistency_regularization.py use a weak and strong augmentation,
    #  force the same pseudo and use strong augementation result as loss
    pass

def train_EDESC(device, i):

    # for whatever reason this function does not take global scope, so this is required for reproducability
    setup_seed(42)

    model = C_EDESC(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain_path=args.pretrain_path).to(device)

    start = time.time()
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
    data = dataset.train

    optimizer = Adam(model.parameters(), lr=args.lr)
    index = dataset.index
    data = torch.Tensor(data).to(device)
    x_bar, hidden = get_initial_value(model, data)

    #passed as decimal number: 0.01 = 1% labeled data used
    label_pct = args.label_usage

    # TODO: chances are that not all classes are within the l_feats selection
    # get random indices throughout the entire dataset
    indices = torch.randperm(len(y))
    print(f'seed: {np.random.get_state()[1][0]}')
    train_size = int(label_pct*len(y)) # only use a percentage of all the indices
    train_indices = indices[:train_size]

    mask_lab = torch.zeros(len(y), dtype=torch.bool) # Create a boolean mask of the same length as y, initialized to False
    mask_lab[train_indices] = True # Set the indices corresponding to labelled samples to True
    u_feats = data[~mask_lab]
    # u_targets = y[~mask_lab]
    l_feats = data[mask_lab]
    l_targets = y[mask_lab]

    random_seed = 42
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=random_seed)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy().reshape(dataset.__len__(), -1))
    print("Initial Cluster Centers: ", y_pred)

    kmeans = SemiSupKMeans(k=args.n_clusters, tolerance=1e-4, max_iterations=300, init='k-means++',
                           n_init=30, random_state=random_seed, n_jobs=-1, pairwise_batch_size=1024,
                           mode=None)

    u_feats = torch.tensor(u_feats, dtype=torch.float32)
    l_feats = torch.tensor(l_feats, dtype=torch.float32)
    l_targets = torch.tensor(l_targets)
    u_feats = u_feats.view(u_feats.size(0), -1)
    l_feats = l_feats.view(l_feats.size(0), -1)

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()

    print(f"kmeans: {accuracy_score(y_pred, y)}")
    print(f"sskmeans unlabelled: {accuracy_score(all_preds[~mask_lab], y[~mask_lab])}")
    print(f"sskmeans labelled: {accuracy_score(all_preds[mask_lab], y[mask_lab])}")
    print(f"sskmeans both: {accuracy_score(all_preds, y)}")

    # TODO: ss-k-means does not increase performance at all, further more it is also inconsistent in being better...
    # y_pred = all_preds

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

        x_bar, s, z, pseudo_logits = model(data)

        ratio = (s > 0.90).sum() / s.shape[0]

        # fancy indexing get the mean prediction of a mini-cluster
        original_center = scatter_mean(src=s, index=torch.tensor(req_c, dtype=torch.int64, device=device), dim=0)
        refined_center = refined_subspace_affinity(original_center)

        y_pred = s.cpu().detach().numpy().argmax(1)
        y_pred_remap, acc, kappa, nmi, ca, mapping = cluster_accuracy(y, y_pred, return_aligned=True)

        # originally this was just ration> max_ratio
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
        y_pred_logits_remap_partial = y_pred_logits_remap[train_indices]
        y_partial = y[train_indices]

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

        # dont forget to apply the permutation here because the latent dim. and actual labels dont necessarily match
        pseudo_loss = model.pseudo_loss_adv2(
            y,
            # pseudo_logits[:, perm]
            s[:, perm]
        )

        s_labeled = s[train_indices]
        y_labeled = torch.tensor(y[train_indices], device=s.device)
        contr_loss = model.supervised_contrastive_loss(s_labeled, y_labeled) #maybe also use/add high conf pseudo labels?
        omega = args.omega
        total_loss  = (total_loss +
                       args.delta * cross_loss +
                       args.epsilon * pseudo_loss +
                       omega * contr_loss
                       )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch_iter % 10 == 0 or epoch_iter == epochs - 1:
            print('Iter {}'.format(epoch_iter), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax), ', Current nmi {:.4f}'.format(nmi),
                  ':Max nmi {:.4f}'.format(nmimax), ', Current kappa {:.4f}'.format(kappa),
                  ':Max kappa {:.4f}'.format(kappa_max))
            print("total_loss", total_loss.data, "reconstr_loss", reconstr_loss.data, "kl_loss", kl_loss.data, "entropy_loss", args.delta * cross_loss.data, "pseudo_loss", args.epsilon * pseudo_loss.data, "contras_loss", omega * contr_loss.data)
            print(ratio)

    end = time.time()
    print('Running time: ', end - start)
    if args.delta != .0:
        print(f"percentage of labeled data used: {label_pct} meaning {len(y_pred_logits_remap_partial)}/{len(y)} used")
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
    parser.add_argument('--delta', default=.0, type=float, help='the weight of the cross_entropy_loss')
    parser.add_argument('--label_usage', default=.001, type=float, help='decimal deciding how much labeled data to be used during training')
    parser.add_argument('--epsilon', default=0.0, type=float, help='the weight of the pseudo_label_loss')
    parser.add_argument('--pseudo_threshold', default=.95, type=float, help='minimum confidence of the pseudo predications to be used')
    parser.add_argument('--topk', default=1, type=int, help="count of pseudo labels to be summed for the threshold to be met")
    parser.add_argument('--omega', default=0.5, type=float, help='the weight of the contrastive_loss')

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
    bestacc = 0
    bestnmi = 0
    best_kappa = 0
    acc_sum = 0
    nmi_sum = 0
    kappa_sum = 0
    rounds = 1
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