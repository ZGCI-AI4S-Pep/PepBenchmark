from __future__ import annotations
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Optional, Dict, Tuple, Any, Sequence


from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

# Optional backend imports
try:
    from pepbenchmark.similarity.fasta import fasta_similarity_batch, SimilarityMethod
except ImportError:
    fasta_similarity_batch = None
    SimilarityMethod = None

try:
    from pepbenchmark.similarity.fp import batch_fingerprint_similarity
except ImportError:
    batch_fingerprint_similarity = None

try:
    from pepbenchmark.similarity.embedding import batch_embedding_similarity
except ImportError:
    batch_embedding_similarity = None

logger = logging.getLogger(__name__)



# =========================
# Basic enums and outputs
# =========================
class InputType(str, Enum):
    SEQUENCE = "sequence"
    FINGERPRINT = "fingerprint"
    EMBEDDING = "embedding"


class MatrixMode(str, Enum):
    FULL = "full"
    BLOCKWISE = "blockwise"


# =========================
# Pairwise similarity matrix only
# =========================
def compute_similarity_matrix(
    data1: List[Union[str, np.ndarray]],
    data2: Optional[List[Union[str, np.ndarray]]] = None,
    input_type: Union[str, InputType] = InputType.SEQUENCE,
    method: Union[str, object] = "levenshtein",
    *,
    processes: Optional[int] = None,
    show_progress: bool = True,
    mode: Union[str, MatrixMode] = MatrixMode.FULL,
    block_size: int = 2000,
    temp_dir: Optional[str] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the pairwise similarity matrix.

    Args:
        data1: First input collection.
        data2: Optional second input collection. When omitted, the function
            computes a self-similarity matrix for ``data1``.
        input_type: Input representation type.
        method: Similarity metric or backend-specific method name.
        processes: Optional worker count for supported backends.
        show_progress: Whether to display progress information.
        mode: Matrix construction mode. Use ``FULL`` for a direct batch
            computation or ``BLOCKWISE`` for lower-memory chunked execution.
        block_size: Block size used when ``mode`` is ``BLOCKWISE``.
        temp_dir: Optional directory for temporary blockwise storage.
        **kwargs: Additional backend-specific keyword arguments.

    Returns:
        A similarity matrix as a ``numpy.ndarray``.
    """
    input_type = InputType(input_type)
    mode = MatrixMode(mode)
    
    # Normalize enum inputs.
    if hasattr(method, 'value'):
        method = method.value
    method = str(method)

    if data2 is None:
        data2 = data1
        is_self_comparison = True
    else:
        # Detect self-comparison by object identity only to avoid ndarray equality ambiguity.
        is_self_comparison = data1 is data2

    n1, n2 = len(data1), len(data2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    if mode == MatrixMode.FULL:
        mat = _compute_full_matrix(
            data1, data2, input_type, method, processes, show_progress, is_self_comparison, **kwargs
        )
    elif mode == MatrixMode.BLOCKWISE:
        mat = _compute_blockwise_matrix(
            data1, data2, input_type, method, block_size, processes, show_progress, temp_dir, **kwargs
        )
    else:
        raise ValueError(f"Unsupported matrix mode: {mode}")

    return mat


def _compute_full_matrix(
    data1: List[Union[str, np.ndarray]],
    data2: List[Union[str, np.ndarray]],
    input_type: InputType,
    method: str,
    processes: Optional[int],
    show_progress: bool,
    is_self_comparison: bool,
    **kwargs: Any
) -> np.ndarray:
    if input_type == InputType.SEQUENCE:
        results = fasta_similarity_batch(
            data1, data2, method=method, processes=processes, show_progress=show_progress, **kwargs
        )
    elif input_type == InputType.FINGERPRINT:
        fp_gen = kwargs.get("fp_gen")
        if fp_gen is None:
            raise ValueError("fp_gen required for fingerprint input_type")
        results = batch_fingerprint_similarity(data1, data2, fp_gen=fp_gen, sim_metric=method)
    else:
        emb1 = np.asarray(data1, dtype=np.float32)
        emb2 = np.asarray(data2, dtype=np.float32)
        results = batch_embedding_similarity(emb1, emb2, metric=method, **kwargs)

    # Build the matrix from the result objects.
    n1, n2 = len(data1), len(data2)
    mat = np.zeros((n1, n2), dtype=np.float32)
    
    for result in results:
        i, j = result.query_index, result.value_index
        if 0 <= i < n1 and 0 <= j < n2 and result.similarity is not None:
            mat[i, j] = float(result.similarity)
    
    np.nan_to_num(mat, copy=False)
    return mat


def _compute_blockwise_matrix(
    data1: List[Union[str, np.ndarray]],
    data2: List[Union[str, np.ndarray]],
    input_type: InputType,
    method: str,
    block_size: int,
    processes: Optional[int],
    show_progress: bool,
    temp_dir: Optional[str],
    **kwargs: Any
) -> np.ndarray:
    n1, n2 = len(data1), len(data2)

    fd, path = tempfile.mkstemp(suffix=".dat", dir=temp_dir or tempfile.gettempdir())
    os.close(fd)
    try:
        mat = np.memmap(path, dtype=np.float32, mode="w+", shape=(n1, n2))
        total_blocks = ((n1 + block_size - 1)//block_size) * ((n2 + block_size - 1)//block_size)

        blocks = [(i, j) for i in range(0, n1, block_size) for j in range(0, n2, block_size)]
        if show_progress:
            blocks = tqdm(blocks, total=total_blocks, desc="Blockwise", unit="block")

        for (i, j) in blocks:
            i_end, j_end = min(i + block_size, n1), min(j + block_size, n2)
            sub1, sub2 = data1[i:i_end], data2[j:j_end]
            if input_type == InputType.SEQUENCE:
                sub_results = fasta_similarity_batch(sub1, sub2, method=method, processes=processes, show_progress=False, **kwargs)
            elif input_type == InputType.FINGERPRINT:
                fp_gen = kwargs.get("fp_gen")
                if fp_gen is None:
                    raise ValueError("fp_gen required for fingerprint input_type")
                sub_results = batch_fingerprint_similarity(sub1, sub2, fp_gen=fp_gen, sim_metric=method)
            else:
                emb1 = np.asarray(sub1, dtype=np.float32)
                emb2 = np.asarray(sub2, dtype=np.float32)
                sub_results = batch_embedding_similarity(emb1, emb2, metric=method, **kwargs)

            for r in sub_results:
                mat[i + r.query_index, j + r.value_index] = float(r.similarity) if r.similarity is not None else 0.0

        mat.flush()
        return np.array(mat)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# =========================
# Dedicated Top-K neighbor retrieval
# =========================

def compute_topk(
    data1: Optional[Sequence[Union[str, np.ndarray]]] = None,
    data2: Optional[Sequence[Union[str, np.ndarray]]] = None,
    *,
    input_type: Union[str, InputType] = InputType.SEQUENCE,
    method: Union[str, object] = "levenshtein",
    topk: int = 10,
    processes: Optional[int] = None,          # Used only in approximate mode (faiss/lsh)
    show_progress: bool = True,
    approx: bool = False,
    precomputed_matrix: Optional[np.ndarray] = None,
    is_self_comparison: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Perform only Top-K selection and do not compute exact similarity internally.

    Usage:
        1) Matrix-driven: provide `precomputed_matrix` (n1 x n2), and `data1/data2` are optional.
             - If `is_self_comparison` is not specified, a square matrix with no `data2`
                 is treated as self-comparison by default.
        2) Approximate retrieval: set `approx=True` (FAISS/MinHash LSH). In this mode,
             `data1` must be provided, and `data2` is optional.

    Parameters:
        - `input_type`/`method` are only used for labeling or approximate retrieval;
            they are not used in matrix-driven mode.
        - `processes` is only meaningful for approximate retrieval.
        - `is_self_comparison` overrides self-comparison inference, which is useful when
            the matrix is square but does not represent self-comparison.
    """
    input_type = InputType(input_type)

    # ---------- Path A: matrix-driven ----------
    if precomputed_matrix is not None:
        M = np.asarray(precomputed_matrix)
        if M.ndim != 2:
            raise ValueError("precomputed_matrix must be a 2D matrix")
        n1, n2 = M.shape

        # Infer or override whether this is self-comparison.
        if is_self_comparison is None:
            # Square matrix + no explicit data2 means self-comparison; otherwise treat as non-self.
            is_self = (n1 == n2) and (data2 is None)
        else:
            is_self = bool(is_self_comparison)

        if topk <= 0 or n1 == 0 or n2 == 0:
            return {i: [] for i in range(n1)}

        actual_k = min(topk + 1, n2) if is_self else min(topk, n2)

        def process_row(i: int) -> Tuple[int, List[Tuple[int, float]]]:
            row = M[i]
            # Mask the diagonal for self-comparison.
            if is_self:
                r = row.copy()
                if i < r.shape[0]:
                    r[i] = -np.inf
                row_use = r
            else:
                row_use = row

            # Use argpartition first, then sort the selected candidates.
            idx = np.argpartition(-row_use, actual_k - 1)[:actual_k]
            pairs = [(int(j), float(row_use[j])) for j in idx if (not is_self or j != i)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:topk]

            return i, pairs

        iterator = range(n1)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Top-{topk}", unit="row")

        # Selection is lightweight when the matrix is already in memory, so serial execution is sufficient by default.
        results = dict(Parallel(n_jobs=1)(
            delayed(process_row)(i) for i in iterator
        ))
        return results

    # ---------- Path B: approximate retrieval ----------
    if approx:
        if data1 is None:
            raise ValueError("approx mode requires data1 (and optionally data2)")
        if data2 is None:
            data2 = data1
            is_self = True
        else:
            is_self = (data1 is data2)

        if topk <= 0 or len(data1) == 0 or len(data2) == 0:
            return {i: [] for i in range(len(data1))}

        if input_type == InputType.EMBEDDING:
            return _topk_embedding_approx(list(data1), list(data2), method, topk, is_self)
        elif input_type in {InputType.SEQUENCE, InputType.FINGERPRINT}:
            return _topk_lsh(list(data1), list(data2), input_type, method, topk, is_self)
        else:
            raise ValueError(f"Unsupported input_type for approx: {input_type}")

    # ---------- No matrix and not approximate: raise an error ----------
    raise ValueError(
        "compute_topk requires either precomputed_matrix or approx=True.\n"
        "Please compute a matrix first with compute_similarity_matrix or a similar method, "
        "or enable approximate retrieval (FAISS/MinHash LSH)."
    )



def _topk_embedding_approx(
    emb1: List[np.ndarray],
    emb2: List[np.ndarray],
    method: str,
    topk: int,
    is_self: bool
) -> Dict[int, List[Tuple[int, float]]]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise ImportError(str(e))

    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    n1, d = emb1.shape
    n2 = emb2.shape[0]

    if method == "cosine":
        faiss.normalize_L2(emb1)
        faiss.normalize_L2(emb2)
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(emb2)
    k = min(topk + (1 if is_self else 0), n2)
    sims, idxs = index.search(emb1, k)

    out: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n1):
        result_list: List[Tuple[int, float]] = []
        for rank in range(k):
            j = int(idxs[i, rank])
            if j < 0:
                continue
            if is_self and i == j:
                continue
            score = float(sims[i, rank]) if method == "cosine" else 1.0 / (1.0 + float(sims[i, rank]))
            result_list.append((j, score))
            if len(result_list) >= topk:
                break
        out[i] = result_list
    return out


def _topk_lsh(
    data1: List[Union[str, np.ndarray]],
    data2: List[Union[str, np.ndarray]],
    input_type: InputType,
    method: str,
    topk: int,
    is_self: bool
) -> Dict[int, List[Tuple[int, float]]]:
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except Exception as e:
        raise ImportError(str(e))

    def to_minhash(obj: Union[str, np.ndarray], num_perm: int = 128, k: int = 5) -> MinHash:
        mh = MinHash(num_perm=num_perm)
        if input_type == InputType.SEQUENCE:
            if isinstance(obj, str) and len(obj) >= k:
                for i in range(len(obj) - k + 1):
                    mh.update(obj[i:i + k].encode("utf8"))
        else:
            if isinstance(obj, str):
                idxs = (i for i, ch in enumerate(obj) if ch == "1")
            else:
                arr = np.asarray(obj).astype(bool).ravel()
                idxs = np.nonzero(arr)[0]
            for idx in idxs:
                mh.update(str(int(idx)).encode("utf8"))
        return mh

    n1, n2 = len(data1), len(data2)
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    mh2 = [to_minhash(s) for s in data2]
    for j, mh in enumerate(mh2):
        lsh.insert(str(j), mh)

    out: Dict[int, List[Tuple[int, float]]] = {}
    iterator = enumerate([to_minhash(s) for s in data1])
    for i, mh in (tqdm(iterator, total=n1, desc=f"Top-{topk}-LSH") if n1 > 1000 else iterator):
        cand = lsh.query(mh)
        sims = [(int(j), mh.jaccard(mh2[int(j)])) for j in cand if not (is_self and int(j) == i)]
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[:topk]
        out[i] = sims
    return out


# =========================
# Utility: convert top-k results to a sparse matrix if needed
# =========================
def build_sparse_matrix_from_topk(
    topk_results: Dict[int, List[Tuple[int, float]]],
    n1: int,
    n2: int,
    *,
    is_self_comparison: bool = True,
    show_progress: bool = False
) -> np.ndarray:
    sim = np.zeros((n1, n2), dtype=np.float32)
    if is_self_comparison:
        np.fill_diagonal(sim, 1.0)
    it = topk_results.items()
    if show_progress:
        it = tqdm(list(it), desc="SparseMatrix")
    for i, result_list in it:
        for j, similarity in result_list:
            if 0 <= i < n1 and 0 <= j < n2:
                v = float(similarity) if similarity is not None else 0.0
                if v > sim[i, j]:
                    sim[i, j] = v
    return sim
if __name__ == "__main__":
    import logging
    import random
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    rng = np.random.default_rng(42)
    random.seed(42)

    # -----------------------------
    # Data builders
    # -----------------------------
    def make_sequences(n=20, length=25, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        return ["".join(random.choice(alphabet) for _ in range(length)) for _ in range(n)]

    def make_fingerprints(n=20, dim=256, density=0.1):
        fps = (rng.random((n, dim)) < density).astype(np.uint8)
        mixed = []
        for i in range(n):
            if i % 2 == 0:
                mixed.append(fps[i])  # ndarray
            else:
                mixed.append("".join("1" if b else "0" for b in fps[i]))
        return mixed

    def make_embeddings(n=32, dim=64):
        return rng.normal(0, 1, size=(n, dim)).astype(np.float32)

    # -----------------------------
    # Assertion helpers
    # -----------------------------
    def assert_square_shape(mat, n):
        assert mat.shape == (n, n), f"Matrix shape should be {(n, n)}, got {mat.shape}"

    def assert_in_range(mat, lo=-1.0, hi=1.0):
        assert np.isfinite(mat).all(), "Matrix contains non-finite values"
        mmin, mmax = float(np.min(mat)), float(np.max(mat))
        assert mmin >= lo - 1e-4 and mmax <= hi + 1e-4, f"Matrix out of range [{lo},{hi}], actual [{mmin},{mmax}]"

    def assert_symmetric(mat, atol=1e-5):
        assert np.allclose(mat, mat.T, atol=atol), "Self-comparison matrix should be approximately symmetric"

    # -----------------------------
    # EMBEDDING
    # -----------------------------
    print("\n[TEST] Embedding - cosine")
    emb = make_embeddings(n=32, dim=64)
    mat_emb = compute_similarity_matrix(
        emb, None, input_type="embedding", method="cosine",
        mode="blockwise", block_size=8, show_progress=False
    )
    assert_square_shape(mat_emb, len(emb))
    assert_in_range(mat_emb, lo=-1.0, hi=1.0)
    assert_symmetric(mat_emb)

    print("[TEST] TopK (matrix-driven) - embedding/cosine")
    tk_emb = compute_topk(
        precomputed_matrix=mat_emb, method="cosine", topk=7, is_self_comparison=True
    )
    assert len(tk_emb) == len(emb)

    print("[TEST] TopK (approx, faiss if available) - embedding/cosine")
    try:
        tk_emb_approx = compute_topk(
            emb, input_type="embedding", method="cosine", topk=7, approx=True, show_progress=False
        )
        assert len(tk_emb_approx) == len(emb)
    except Exception as e:
        print(f"  -> Skip approximate mode (faiss unavailable): {e}")

    # -----------------------------
    # SEQUENCE
    # -----------------------------
    try:
        print("\n[TEST] Sequence - levenshtein")
        seqs = make_sequences(n=16, length=20)
        mat_seq = compute_similarity_matrix(
            seqs, None, input_type="sequence", method="levenshtein",
            mode="blockwise", block_size=6, show_progress=False
        )
        assert_square_shape(mat_seq, len(seqs))
        assert np.isfinite(mat_seq).all()

        print("[TEST] TopK (matrix-driven) - sequence/levenshtein")
        tk_seq = compute_topk(
            precomputed_matrix=mat_seq, method="levenshtein", topk=5, is_self_comparison=True
        )

        print("[TEST] TopK (approx, datasketch if available) - sequence/levenshtein")
        try:
            tk_seq_approx = compute_topk(
                seqs, input_type="sequence", method="levenshtein", topk=5, approx=True, show_progress=False
            )
        except Exception as e:
            print(f"  -> Skip approximate mode (datasketch unavailable): {e}")
    except Exception as e:
        print(f"  -> Skip sequence test: {e}")

    # -----------------------------
    # FINGERPRINT
    # -----------------------------
    def simple_fp_gen(x):
        if isinstance(x, str):
            return np.frombuffer(x.encode("ascii"), dtype=np.uint8) == ord("1")
        return np.asarray(x).astype(bool)

    try:
        print("\n[TEST] Fingerprint - jaccard")
        fps = make_fingerprints(n=20, dim=256, density=0.08)
        mat_fp = compute_similarity_matrix(
            fps, None, input_type="fingerprint", method="jaccard",
            mode="blockwise", block_size=8, show_progress=False,
            fp_gen=simple_fp_gen
        )
        assert_square_shape(mat_fp, len(fps))
        assert_in_range(mat_fp, lo=0.0, hi=1.0)
        assert_symmetric(mat_fp, atol=1e-6)

        print("[TEST] TopK (matrix-driven) - fingerprint/jaccard")
        tk_fp = compute_topk(
            precomputed_matrix=mat_fp, method="jaccard", topk=6, is_self_comparison=True
        )

        print("[TEST] TopK (approx, datasketch if available) - fingerprint/jaccard")
        try:
            tk_fp_approx = compute_topk(
                fps, input_type="fingerprint", method="jaccard", topk=6, approx=True,
                fp_gen=simple_fp_gen, show_progress=False
            )
        except Exception as e:
            print(f"  -> Skip approximate mode (datasketch unavailable): {e}")
    except Exception as e:
        print(f"  -> Skip fingerprint test: {e}")

    # -----------------------------
    # CROSS SET
    # -----------------------------
    print("\n[TEST] Cross set - embedding/cosine")
    emb_q = make_embeddings(n=12, dim=64)
    emb_v = make_embeddings(n=15, dim=64)
    mat_cross = compute_similarity_matrix(
        emb_q, emb_v, input_type="embedding", method="cosine",
        mode="blockwise", block_size=6, show_progress=False
    )
    assert mat_cross.shape == (len(emb_q), len(emb_v))

    print("[TEST] Cross set TopK (matrix-driven)")
    tk_cross = compute_topk(
        precomputed_matrix=mat_cross, method="cosine", topk=5, is_self_comparison=False
    )
    assert len(tk_cross) == len(emb_q)

    print("\n✅ All tests completed.")
