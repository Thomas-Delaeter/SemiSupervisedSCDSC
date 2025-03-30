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

from my_knn import get_initial_value
from Auto_encoder import Ae
from getdata import Load_my_Dataset
from utils import cluster_accuracy
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


# 设置随机数种子
# setup_seed(42)


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

        # Pseudo-Graph Module
        # TODO: batch_size
        # Makes it O(n**2) again, so best to avoid?
        # proposed method uses small set of samples to train network
        # Not end to end trainable anymore
        # self.Coef = Parameter(torch.Tensor(batch_size,batch_size))
        # nn.init.constant_(self.Coef, 1e-5)

        # Pseudo-Label Module
        latent_dim = n_z * 7 * 7
        self.pseudo_classifier = nn.Linear(latent_dim, n_clusters)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
            self.ae.load_state_dict(torch.load(self.pretrain_path,map_location=device))
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path,map_location=device))
        print('Load pre-trained model from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)
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

        # Pseudo Label
        pseudo_logits = self.pseudo_classifier(z)

        return x_bar, s, z, pseudo_logits

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

    def total_loss_semi(self, x, x_bar, center, target, dim, n_clusters, s, index, y, y_pred):
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


        # crossentropy
        cross_loss = F.cross_entropy(y_pred, torch.from_numpy(y).long()) if not y_pred.numel() == 0 else torch.tensor(0)

        total_loss = reconstr_loss + loss_d1 + loss_d2 + args.alpha * kl_loss + args.beta * loss_smooth + cross_loss


        return total_loss, reconstr_loss, kl_loss, loss_d1, loss_d2, cross_loss

    def total_loss_semi_pseudo(self, x, x_bar, center, target, dim, n_clusters, s, index, y, y_limited, y_pred, pseudo_logits):
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

        # Pseudo Label Loss
        # Samples with high confidence (e.g. max probability > 0.8, like PSSC framework)
        max_probs, _ = torch.max(F.softmax(pseudo_logits, dim=1), dim=1)
        confident_mask = (max_probs > args.pseudo_threshold)
        if confident_mask.sum() > 0 and y is not None:
            pseudo_loss = F.cross_entropy(pseudo_logits[confident_mask], torch.from_numpy(y[confident_mask.cpu().numpy()]).long().to(device))
        else:
            if args.epsilon != .0:
                print("no pseudo labels were valuable enough to contribute")
            global pseudo_miss #TODO: prettify this because global variables arent really clean code
            pseudo_miss+=1
            pseudo_loss = torch.tensor(0.0, device=device)

        # Cross-Entropy Loss
        cross_loss = F.cross_entropy(y_pred, torch.from_numpy(y_limited).long()) if not y_pred.numel() == 0 else torch.tensor(0)

        total_loss = reconstr_loss + loss_d1 + loss_d2 + args.alpha * kl_loss + args.beta * loss_smooth+ args.delta * cross_loss + args.epsilon * pseudo_loss


        return total_loss, reconstr_loss, kl_loss, loss_d1, loss_d2, cross_loss, pseudo_loss

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
            loss = F.mse_loss(x_bar, x) #+ F.cross_entropy()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))



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

    # utilze finch to generate big groups
    from finch import FINCH
    data = data.reshape(data.shape[0], -1)
    c, num_clust, req_c1 = FINCH(data)
    req_c = c[:, args.hierarchy]

    model.pretrain(f'original_weight/{args.dataset}.pkl')
    # model.pretrain('')
    data = dataset.train
    optimizer = Adam(model.parameters(), lr=args.lr)
    index = dataset.index
    data = torch.Tensor(data).to(device)
    x_bar, hidden = get_initial_value(model, data)

    #passed as decimal number
    label_pct = args.label_usage #0.01

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

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    # preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    # mask = mask_cls[~mask_lab].numpy()
    # mask = mask.astype(bool)

    # -----------------------
    # EVALUATE
    # -----------------------

    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=y, y_pred=all_preds, mask=mask_lab.numpy(),
    #                                                 eval_funcs=['v1', 'v2'],
    #                                                 save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    print(f"kmeans: {accuracy_score(y_pred, y)}")
    print(f"sskmeans unlabelled: {accuracy_score(all_preds[~mask_lab], y[~mask_lab])}")
    print(f"sskmeans labelled: {accuracy_score(all_preds[mask_lab], y[mask_lab])}")
    print(f"sskmeans both: {accuracy_score(all_preds, y)}")

    y_pred = all_preds # transition to ss-k-means

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

    for epoch in range(epochs):

        x_bar, s, z, pseudo_logits = model(data)

        ratio = (s > 0.90).sum() / s.shape[0]

        # fancy indexing get the mean prediction of a mini-cluster
        original_center = scatter_mean(src=s, index=torch.tensor(req_c, dtype=torch.int64, device=device), dim=0)
        refined_center = refined_subspace_affinity(original_center)

        y_pred = s.cpu().detach().numpy().argmax(1)
        y_best, acc, kappa, nmi, ca, mapping = cluster_accuracy(y, y_pred, return_aligned=True)

        if ratio > max_ratio:
            accmax = acc
            kappa_max = kappa
            nmimax = nmi
            ca_max = ca
            max_ratio = ratio


        # since y_best is a remap, we add the mapping to the logits.
        logits = s.cpu()#.detach().numpy() #is detaching allowed?

        num_classes = logits.size(1)
        # Create permutation index array
        perm = [0] * num_classes
        for old_index, new_index in mapping.items():
            perm[new_index] = old_index

        # Convert to tensor and ensure it's on the same device as logits
        perm = torch.tensor(perm, dtype=torch.long, device=logits.device)

        # Apply permutation to logits
        y_best_logits = logits[:, perm]

        # Train test splitting as an arbitrary way of limiting data
        y_best_logits_limited = y_best_logits[train_indices]
        y_limited = y[train_indices]

        total_loss, reconstr_loss, kl_loss, loss_d1, loss_d2, entropy_loss, pseudo_loss = model.total_loss_semi_pseudo(
                                                                                    x=data,
                                                                                    x_bar=x_bar,
                                                                                    center=original_center,
                                                                                    target=refined_center,
                                                                                    dim=args.d,
                                                                                    n_clusters=args.n_clusters,
                                                                                    s=s,
                                                                                    index=index,
                                                                                    y = y,
                                                                                    y_limited = y_limited,
                                                                                    y_pred = y_best_logits_limited,
                                                                                    pseudo_logits=pseudo_logits
                                                                                    )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax), ', Current nmi {:.4f}'.format(nmi),
                  ':Max nmi {:.4f}'.format(nmimax), ', Current kappa {:.4f}'.format(kappa),
                  ':Max kappa {:.4f}'.format(kappa_max))
            print("total_loss", total_loss.data, "reconstr_loss", reconstr_loss.data, "kl_loss", kl_loss.data, "entropy_loss", entropy_loss.data, "pseudo_loss", pseudo_loss.data)
            print(ratio)

    end = time.time()
    print('Running time: ', end - start)
    print(f"percentage of labeled data used: {label_pct} meaning {len(y_best_logits_limited)}/{len(y)} used")
    return accmax, nmimax, kappa_max, ca_max




if __name__ == "__main__":
    import datetime

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
    parser.add_argument('--dataset', type=str, default='trento')
    parser.add_argument('--device_index', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default='./tmp/flood/pre.pkl')
    parser.add_argument('--alpha', default=3, type=float, help='the weight of kl_loss')
    parser.add_argument('--beta', default=8, type=float, help='the weight of local_loss')
    parser.add_argument('--gama', default=0.03, type=float, help='the weight of non_local_loss')
    parser.add_argument('--delta', default=.0, type=float, help='the weight of the cross_entropy_loss')
    parser.add_argument('--label_usage', default=.05, type=float, help='decimal deciding how much labeled data to be used during training')
    parser.add_argument('--epsilon', default=.0, type=float, help='the weight of the pseudo_label_loss')
    parser.add_argument('--pseudo_threshold', default=.8, type=float, help='minimum confidence of the pseudo predications to be used')

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