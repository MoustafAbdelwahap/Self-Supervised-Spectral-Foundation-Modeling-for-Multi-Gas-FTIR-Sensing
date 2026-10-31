"""Extract and visualize latent representations using UMAP or t-SNE."""

import logging
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch

from sfm.data.dataset import create_dataloaders
from sfm.models.mae import SpectralMAE
from sfm.training.utils import get_device, load_checkpoint

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: SpectralMAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 5000,
) -> np.ndarray:
    """Extract mean-pooled latent vectors from unmasked spectra using the encoder.

    Args:
        model: Trained SpectralMAE model.
        dataloader: DataLoader yielding spectra tensors.
        device: Device to run forward pass on.
        num_samples: Maximum number of sample embeddings to extract.

    Returns:
        Numpy array of shape (N, encoder_dim).
    """
    model.eval()
    embeddings_list = []
    collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)

            # Encoded shape: (batch_size, num_patches, encoder_dim)
            tokens = model.encode(batch)
            # Global average pooling across patches -> (batch_size, encoder_dim)
            pooled = tokens.mean(dim=1).cpu().numpy()

            embeddings_list.append(pooled)
            collected += len(pooled)
            if collected >= num_samples:
                break

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings[:num_samples]


def reduce_dimensions(
    embeddings: np.ndarray, method: str = "umap", seed: int = 42
) -> np.ndarray:
    """Project high-dimensional spectral embeddings to 2D manifold.

    Args:
        embeddings: Array of shape (N, dim).
        method: Dimensionality reduction technique ('umap' or 'tsne').
        seed: Random seed for projection stability.

    Returns:
        Array of shape (N, 2).
    """
    method = method.lower().strip()
    logger.info("Projecting %d embeddings to 2D via %s (seed=%d)", len(embeddings), method, seed)

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed, min_dist=0.3, n_neighbors=15)
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not available, falling back to scikit-learn t-SNE")
            method = "tsne"

    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(5.0, len(embeddings) / 4.0))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    return tsne.fit_transform(embeddings)


def plot_embeddings(
    coords_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Embeddings (SFM)",
    output_path: Optional[str] = None,
) -> None:
    """Generate high-resolution scatter plot of 2D embeddings.

    Args:
        coords_2d: 2D coordinates array of shape (N, 2).
        labels: Optional cluster or gas category labels for coloring.
        title: Plot title.
        output_path: Filepath where PNG should be saved.
    """
    plt.figure(figsize=(9, 7))
    if labels is not None and len(labels) == len(coords_2d):
        scatter = plt.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
            s=12,
            edgecolors="none",
        )
        plt.colorbar(scatter, label="Gas Label / Cluster")
    else:
        plt.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c="#1f77b4",
            alpha=0.6,
            s=12,
            edgecolors="none",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Saved embedding plot to %s", output_path)
    else:
        plt.show()


def visualize_embeddings(
    checkpoint_path: str,
    data_dir: str,
    output_path: str,
    method: str = "umap",
    num_samples: int = 5000,
    seed: int = 42,
) -> None:
    """Orchestrate embedding extraction, dimensionality reduction, and rendering.

    Args:
        checkpoint_path: Path to trained SpectralMAE checkpoint.
        data_dir: Directory containing spectra.
        output_path: Target PNG image output path.
        method: 'umap' or 'tsne'.
        num_samples: Number of samples to visualize.
        seed: Random state.
    """
    device = get_device("auto")
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]

    # Determine channels
    wn_path = os.path.join(data_dir, "wavenumbers.npy")
    num_channels = len(np.load(wn_path)) if os.path.exists(wn_path) else 200

    from sfm.training.pretrain import create_model

    model = create_model(config, num_channels=num_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    loaders = create_dataloaders(data_dir=data_dir, batch_size=128, num_workers=2, seed=seed)
    dataloader = loaders.get("pretrain") or loaders.get("train") or loaders.get("test")
    if dataloader is None:
        raise FileNotFoundError(f"No valid spectral data found in {data_dir}")

    embeddings = extract_embeddings(model, dataloader, device, num_samples=num_samples)
    coords_2d = reduce_dimensions(embeddings, method=method, seed=seed)
    plot_embeddings(
        coords_2d,
        title=f"Latent Spectral Embedding Clusters ({method.upper()})",
        output_path=output_path,
    )

