import math
from typing import List, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def embedding_similarity(
    emb1: Union[np.ndarray, List],
    emb2: Union[np.ndarray, List] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute embedding similarity matrix.

    Args:
        emb1: Embeddings (n1, d), or a single vector (d,).
        emb2: Embeddings (n2, d). If None, self-comparison is performed.
        metric: "cosine", "euclidean", or "dot".

    Returns:
        np.ndarray of shape (n1, n2).
    """
    metric = metric.lower()
    if metric not in {"cosine", "euclidean", "dot"}:
        raise ValueError(f"Unsupported metric: {metric}")

    emb1 = np.atleast_2d(np.asarray(emb1, dtype=np.float32))
    if emb2 is None:
        emb2 = emb1
    else:
        emb2 = np.atleast_2d(np.asarray(emb2, dtype=np.float32))

    d = emb1.shape[1]

    if metric == "cosine":
        return ((cosine_similarity(emb1, emb2) + 1.0) / 2.0).astype(np.float32)
    elif metric == "euclidean":
        dist_mat = euclidean_distances(emb1, emb2)
        max_dist = math.sqrt(4.0 * d)
        return np.maximum(0.0, 1.0 - dist_mat / max_dist).astype(np.float32)
    else:  # dot
        n1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-12)
        n2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-12)
        return ((np.dot(n1_norm, n2_norm.T) + 1.0) / 2.0).astype(np.float32)


if __name__ == "__main__":
    import numpy as _np
    e1 = _np.random.randn(640).astype(_np.float32)
    e2 = _np.random.randn(640).astype(_np.float32)
    for m in ["cosine", "euclidean", "dot"]:
        sim = float(embedding_similarity(e1, e2, metric=m)[0, 0])
        print(f"[{m}] similarity: {sim:.4f}")

    emb1 = _np.random.randn(3, 128).astype(_np.float32)
    emb2 = _np.random.randn(2, 128).astype(_np.float32)
    mat = embedding_similarity(emb1, emb2, metric="cosine")
    print("Batch similarity matrix shape:", mat.shape)
    print(mat)
