from collections import defaultdict
import numpy as np
import torch


def seperate(Z, y_pred, n_clusters):
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate



def Initialization_D(Z, y_pred, n_clusters, d):
    # 将隐空间特征Z按照簇来进行分类
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([Z.shape[1], n_clusters * d])
    print("Initialize D")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100,random_state=42)
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        svd.fit(Z_seperate[i])
        val = svd.singular_values_
        u = (svd.components_).transpose()
        if u.shape[1]>=d:
            U[:, i * d:(i + 1) * d] = u[:, 0:d]
        else:
            print("Not balance")
            U[:, i * d:(i * d)+u.shape[1]] = u[:, :]

    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D
