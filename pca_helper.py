from sklearn.decomposition import PCA


def pca_helper_func(components, input_data):
    pca = PCA(n_components=components, random_state=42)
    print(f"Created pca obj.\nFitting to input data: {input_data.shape} -> {components}")
    pca.fit(input_data)
    print("Now Transforming.")
    output = pca.transform(input_data)
    print("Done.")
    return output


import torch
import numpy as np


def pca_transform(x, n_components: int):
    """
    Performs PCA on input data x, reducing its dimensionality to n_components.
    This function computes the mean automatically, centers the data,
    computes the principal components using SVD, and projects x onto these components.

    Args:
        x (np.ndarray or torch.Tensor): Input data of shape (n_samples, n_features).
        n_components (int): The desired number of principal components.

    Returns:
        torch.Tensor: Transformed data of shape (n_samples, n_components).
        torch.Tensor: The computed mean (shape: (n_features,)).
        torch.Tensor: The principal components (shape: (n_components, n_features)).
    """
    # Convert input to torch.Tensor if needed.
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))

    # Compute the mean and center the data.
    mean = x.mean(dim=0)
    x_centered = x - mean

    # Compute SVD of the centered data.
    # Note: torch.linalg.svd returns U, S, Vh (where Vh is the conjugate transpose of V).
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)

    # The principal components are the rows of Vh (each row is a component).
    # We take the first n_components rows.
    principal_components = Vh[:n_components, :]

    # Project the data onto the selected principal components.
    # Since Vh has shape (n_features, n_components) after transposition,
    # we can compute the projection as:
    x_transformed = torch.matmul(x_centered, principal_components.t())

    return x_transformed#, mean, principal_components


# Example usage:
if __name__ == "__main__":
    # Suppose we have 30214 samples with 2312 features.
    # For demonstration, we generate random data.
    x_np = np.random.randn(30214, 2312).astype(np.float32)

    # We want to reduce the dimensionality to 300 components.
    n_components = 300

    x_transformed, mean, components = pca_transform(x_np, n_components)
    print("Transformed shape:", x_transformed.shape)  # Expected: torch.Size([30214, 300])
