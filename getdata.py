import numpy as np
import torch
from sklearn.preprocessing import scale, normalize, minmax_scale, robust_scale
from osgeo import gdal

def get_data(img_path, label_path):
    if img_path[-3:] == 'tif':
        img_data = gdal.Open(img_path).ReadAsArray()
        label_data = gdal.Open(label_path).ReadAsArray()
        img_data = np.transpose(img_data, [1, 2, 0])
        return img_data, label_data
    elif img_path[-3:] == 'mat':
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        img_keys = img_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']

        if label_path is not None:
            gt_mat = sio.loadmat(label_path)
            gt_keys = gt_mat.keys()
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float32'), gt_mat.get(gt_key[0]).astype('int8')
        return img_mat.get(img_key[0]).astype('float32'), img_mat.get(img_key[1]).astype('int8')



def get_data_patch(data, patch_size):
    patch_w = patch_size[0]
    patch_h = patch_size[1]

    # Get the size of the padding
    pad_h = int((patch_h - 1) / 2)
    pad_w = int((patch_w - 1) / 2)

    res = np.zeros((data.shape[0], data.shape[1], patch_w, patch_h, data.shape[2]))

    # Get the padded image
    data_ = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'edge')

    for i in range(pad_h, data.shape[0] - pad_h):
        for j in range(pad_w, data_.shape[1] - pad_w):
            patch = data_[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1, :]
            res[i - pad_h, j - pad_w, :, :, :] = patch
    return res


def get_patch(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """
    :param array: Data to be processed
    :param window: Size of the window. If it is a single value, it corresponds to the length of the last dimension. If it is a tuple, it corresponds to the dimensions and size of the block
    :param asteps:
    :param wsteps:
    :param axes: Corresponds to each dimension of the window
    :param toend:
    :return:
    """
    array = np.asarray(array)  # Redundant?
    orig_shape = np.asarray(array.shape)

    # Convert scalar to array, while array remains unchanged
    window = np.atleast_1d(window).astype(int)

    # Determine the dimensions and size of each dimension of the block
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axes] = size
        window = w

    # Clearly, window is a one-dimensional scalar
    if window.ndim > 1:
        raise ValueError("window must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("The length of each dimension must be greater than one.")
    # The purpose of this function is to divide the original large matrix into smaller blocks, so the size of each block must be smaller than the original matrix
    if len(array.shape) < len(window):
        raise ValueError("The dimensions of window must be smaller or equal to the original")

    _asteps = np.ones_like(orig_shape)
    # asteps is probably the step size outside the window
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError('asteps must be a 1D list of step sizes for each dimension of the window')
        if len(asteps) > array.ndim:
            raise ValueError('block dims bigger than array dims')
        # What does this mean? Debugging
        _asteps[-len(asteps):] = asteps
        if np.any(asteps < 1):
            raise ValueError("Step sizes must be greater than or equal to 1")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        # This is the step size within the window, i.e., the sampling step size for each dimension
        if wsteps.shape != window.shape:
            raise ValueError("Dimension error")
        if np.any(wsteps <= 0):
            raise ValueError("All step sizes must be greater than 0")
        _wsteps[:] = wsteps
        # Steps must be at least 1
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    # The size of the window must be less than or equal to the corresponding dimension of the original matrix
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError('window*wsteps larger than array in at least one dimension')

    new_shape = orig_shape
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window)] = window
        _window = _.copy()
        _[-len(window)] = new_strides
        _new_strides = _
        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def get_HSI_patches(x, gt, ksize, stride=(1, 1), padding='reflect', is_index=False, is_labeled=True):
    """
    :param x: input data
    :param gt: label data
    :param ksize: kernel size
    :param stride: stride value
    :param padding: padding mode
    :param is_index:
    :param is_labeled:
    :return:
    """
    # np.ceil rounds up
    new_height = np.ceil(x.shape[0] / stride[0])
    new_width = np.ceil(x.shape[1] / stride[1])
    band = x.shape[2]

    pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
    pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]

    pad_top = int(pad_needed_height / 2)
    pad_down = int(pad_needed_height - pad_top)
    pad_left = int(pad_needed_width / 2)
    pad_right = int(pad_needed_width - pad_left)

    # 3D images need to be padded in each dimension
    x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
    # Does the label also need padding?
    gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)

    n_row, n_clm, n_band = x.shape

    x = np.reshape(x, (n_row, n_clm, n_band))
    y = np.reshape(gt, (n_row, n_clm))

    ksize = (ksize[0], ksize[1])

    x_patches = get_patch(x, ksize, axes=(1, 0))
    y_patches = get_patch(y, ksize, axes=(1, 0))

    i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[1] - 1) // 2)
    nonzero_index = np.where(y_patches[:, :, i_1, i_2] > 0)
    y_patches_non = y_patches + 1
    all_index = y_patches_non[:, :, i_1, i_2].nonzero()
    if is_labeled is False:
        return x_patches.reshape(
            [x_patches.shape[0] * x_patches.shape[1], x_patches.shape[2], x_patches.shape[3], x_patches.shape[4]]), \
               y_patches[:, :, i_1, i_2].reshape([x_patches.shape[0] * x_patches.shape[1]]), all_index
    x_patches_nonzero = x_patches[nonzero_index]
    y_patches_nonzero = (y_patches[:, :, i_1, i_2])[nonzero_index]

    x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 2, 3, 1])
    if is_index is True:
        return x_patches_nonzero, y_patches_nonzero, nonzero_index

    y_patches_nonzero = standardize_label(y_patches_nonzero)

    print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))

    y_patches_nonzero = y_patches_nonzero.flatten()
    return x_patches_nonzero, y_patches_nonzero, nonzero_index


def standardize_label(y):
    """
    Standardize class labels into 0-k
    :param y:
    :return:
    """
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


class Load_my_Dataset():
    def __init__(self, image_path, label_path):
        X, Y = get_data(image_path, label_path)
        n_row, n_column, n_band = X.shape

        # perform PCA
        from sklearn.decomposition import PCA
        # X = scale(X.reshape(n_row * n_column, n_band))
        # # np.savetxt('X.txt', X, fmt='%d', delimiter=' ')
        # n_components = 8
        # pca = PCA(n_components)
        # X = pca.fit_transform(X.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_components))
        #
        # x_patches, y_patches, index = get_HSI_patches(x=X, gt=Y, ksize=(17, 17), is_labeled=True)
        # x_patches = scale(x_patches.reshape(x_patches.shape[0], -1))
        #
        # n_components = 300
        # pca = PCA(n_components)
        # print("step1ok")
        # x_patches = pca.fit_transform(x_patches)
        X = scale(X.reshape(n_row * n_column, n_band))
        n_components = 8
        pca = PCA(n_components, random_state=42)
        X = X.reshape(n_row*n_column, n_band)

        print(f'X shape before pca 8comp. : {X.shape}')
        X = pca.fit_transform(X)
        print(f'X shape after pca 8comp. : {X.shape}')

        X = X.reshape((n_row, n_column, n_components))
        print(f'X shape reverted back to: {X.shape}')

        x_patches, y_patches, index = get_HSI_patches(x=X, gt=Y, ksize=(17, 17), is_labeled=True)
        x_patches = x_patches.reshape(x_patches.shape[0], -1)
        x_patches = scale(x_patches)

        n_components = 30
        #pca2 = PCA(n_components=n_components, random_state=42)
        print("step1ok")

        print(f"x_patches Shape: {x_patches.shape}")
        #x_patches_new = np.loadtxt('XL.txt', delimiter=' ')
        #x_patches = pca2.fit_transform(x_patches)
        from pca_helper import pca_helper_func,pca_transform
        x_patches2=pca_transform(x_patches,300)
        x_patches=pca_helper_func(300,x_patches)

        print('normalizing patches')
        x_patches = normalize(X=x_patches)
        x_train, _, _ = get_HSI_patches(x=X, gt=Y, ksize=(7, 7), is_labeled=True)
        x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
        n_samples, n_row, n_col, n_channel = x_train.shape
        x_train = scale(x_train.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))

        x_patches_pre, y_patches_pre, _ = get_HSI_patches(x=X, gt=Y, ksize=(7, 7), is_labeled=False)
        x_patches_pre = np.transpose(x_patches_pre, [0, 2, 3, 1])
        n_samples, n_row, n_col, n_channel = x_patches_pre.shape
        x_patches_pre = scale(x_patches_pre.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        x_patches_pre = np.transpose(x_patches_pre, axes=(0, 3, 1, 2))

        y_patches = y_patches.reshape(-1)
        self.x, self.y, self.index, self.train, self.train_pre, self.y_patches_pre = x_patches, y_patches, index, x_train, x_patches_pre, y_patches_pre.reshape(-1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.train[idx])), torch.from_numpy(
            np.array(self.index[0][idx])), torch.from_numpy(np.array(self.index[1][idx]))
