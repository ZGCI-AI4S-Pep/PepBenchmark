"""Sequence-similarity backends for peptide FASTA strings."""

from enum import Enum
from typing import Any, Dict, List, Optional
from collections import Counter
import math

import numpy as np

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    import parasail
except ImportError:
    parasail = None


# =========
# Configuration and utilities
# =========
AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO-"
MATRICES: Dict[str, Any] = {}
if parasail is not None:
    MATRICES = {
        "blosum62": parasail.blosum62,
        "blosum80": parasail.blosum80,
        "blosum50": parasail.blosum50,
        "pam250": parasail.pam250,
    }

DEFAULT_MATRIX_NAME = "blosum62"
DEFAULT_MATRIX = MATRICES.get(DEFAULT_MATRIX_NAME)


# --- Utility functions ---
def _prep(seq: str) -> str:
    return (seq or "").upper()

def _require_parasail() -> None:
    if parasail is None:
        raise ImportError(
            "Alignment-based sequence similarity requires the optional "
            "dependency 'parasail'."
        )


def _safe_get_matrix(name: Optional[str]) -> Any:
    _require_parasail()
    if not name:
        return DEFAULT_MATRIX
    key = name.lower()
    return MATRICES.get(key, DEFAULT_MATRIX)

def _identity_from_trace(tb) -> tuple[int, int, float]:
    # tb.query / tb.ref are aligned strings that may contain gaps.
    q, r = tb.query, tb.ref
    aln_len = len(q)
    if aln_len == 0:
        return 0, 0, 1.0
    matches = sum(1 for a, b in zip(q, r) if a == b and a != '-' and b != '-')
    identity = matches / aln_len
    return matches, aln_len, identity

def _lcs_len(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def _cosine_from_counter(c1: Counter, c2: Counter) -> float:
    if not c1 and not c2:
        return 1.0
    # dot
    keys = set(c1.keys()) | set(c2.keys())
    dot = sum(c1[k] * c2[k] for k in keys)
    # norms
    n1 = math.sqrt(sum(v*v for v in c1.values()))
    n2 = math.sqrt(sum(v*v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _levenshtein_distance(a: str, b: str) -> int:
    if Levenshtein is not None:
        return Levenshtein.distance(a, b)

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def _jaro_winkler_similarity(a: str, b: str, prefix_scale: float = 0.1) -> float:
    if Levenshtein is not None:
        return Levenshtein.jaro_winkler(a, b)

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    len_a, len_b = len(a), len(b)
    max_dist = max(len_a, len_b) // 2 - 1
    max_dist = max(0, max_dist)

    a_matches = [False] * len_a
    b_matches = [False] * len_b
    matches = 0

    for i, char_a in enumerate(a):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len_b)
        for j in range(start, end):
            if b_matches[j] or char_a != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    j = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[j]:
            j += 1
        if a[i] != b[j]:
            transpositions += 1
        j += 1

    transpositions /= 2
    jaro = (
        (matches / len_a) + (matches / len_b) + ((matches - transpositions) / matches)
    ) / 3

    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        prefix += 1

    return jaro + prefix * prefix_scale * (1 - jaro)


class SimilarityMethod(Enum):
    """Supported sequence-similarity methods."""

    SLIDING_WINDOW = "sliding_window"
    LEVENSHTEIN = "levenshtein"
    JACCARD_KMER = "jaccard_kmer"
    DICE_KMER = "dice_kmer"
    COSINE_KMER = "cosine_kmer"          # newly added
    JARO_WINKLER = "jaro_winkler"
    LCS = "lcs"
    NW_LINEAR = "nw_linear"              # actually affine-gap alignment (name kept for compatibility)
    NW_BLOSUM62 = "nw_blosum62"
    SMITH_WATERMAN = "smith_waterman"
    HAMMING = "hamming"                  # newly added (equal-length only)

    @classmethod
    def list_methods(cls) -> list[str]:
        """Return all supported method names."""
        return [m.value for m in cls]


# =========================
# Internal single-pair core
# =========================
def _fasta_similarity_single(
    seq1: str,
    seq2: str,
    method: str = "levenshtein",
    **kwargs
) -> float:
    """Compute similarity for a single sequence pair. Internal use only."""
    return float(fasta_similarity([seq1], [seq2], method=method, **kwargs)[0, 0])


# =========================
# Unified batch interface
# =========================
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _init_worker(seqs1, seqs2, method, kwargs):
    global _S1, _S2, _METHOD, _KWARGS
    _S1, _S2, _METHOD, _KWARGS = seqs1, seqs2, method, kwargs


def _worker(pair):
    i, j = pair
    sim = _compute_single(_S1[i], _S2[j], method=_METHOD, **_KWARGS)
    return i, j, sim


def _compute_single(seq1: str, seq2: str, method: str, **kwargs) -> float:
    method = method.lower()
    s1, s2 = _prep(seq1), _prep(seq2)

    if method == "levenshtein":
        dist = _levenshtein_distance(s1, s2)
        align_len = max(len(s1), len(s2))
        sim = 1.0 - dist / align_len if align_len > 0 else 1.0
        return float(sim)

    elif method == "jaro_winkler":
        sim = _jaro_winkler_similarity(s1, s2)
        return float(sim)

    elif method == "hamming":
        if len(s1) != len(s2):
            # Hamming distance is only meaningful for equal-length sequences.
            # For unequal lengths, return 0 and record the compared length.
            return 0.0
        mismatches = sum(a != b for a, b in zip(s1, s2))
        sim = 1.0 - mismatches / len(s1) if len(s1) else 1.0
        return float(sim)

    elif method == "lcs":
        lcs_len = _lcs_len(s1, s2)
        align_len = max(len(s1), len(s2))
        sim = lcs_len / align_len if align_len > 0 else 1.0
        return float(sim)
    elif method in ("nw_linear", "nw_affine", "nw_blosum62"):
        _require_parasail()
        matrix_name = kwargs.get("matrix", "blosum62") if method != "nw_blosum62" else "blosum62"
        matrix = _safe_get_matrix(matrix_name)
        gap_open = int(kwargs.get("gap_open", 7))
        gap_extend = int(kwargs.get("gap_extend", 1))
        res = parasail.nw_trace_scan_16(s1, s2, gap_open, gap_extend, matrix)
        matches, aln_len, identity = _identity_from_trace(res.traceback)
        sim = identity   # Use identity as the single similarity score.
        return float(sim)

    elif method == "smith_waterman":
        _require_parasail()
        matrix = _safe_get_matrix(kwargs.get("matrix", "blosum62"))
        gap_open = int(kwargs.get("gap_open", 7))
        gap_extend = int(kwargs.get("gap_extend", 1))
        res = parasail.sw_trace_scan_16(s1, s2, gap_open, gap_extend, matrix)
        matches, aln_len, identity = _identity_from_trace(res.traceback)
        sim = identity
        return float(sim)

    elif method == "sliding_window":
        best = 0
        n, m = len(s1), len(s2)
        if n == 0 and m == 0:
            sim = 1.0
            return float(sim)
        for off in range(-m + 1, n):
            start, end = max(0, off), min(n, off + m)
            matches = sum(s1[i] == s2[i - off] for i in range(start, end))
            best = max(best, matches)
        sim = best / max(n, m)
        return float(sim)

    elif method == "jaccard_kmer":
        k = max(1, int(kwargs.get("k", 3)))
        S1 = {s1[i:i+k] for i in range(len(s1) - k + 1)} if len(s1) >= k else set()
        S2 = {s2[i:i+k] for i in range(len(s2) - k + 1)} if len(s2) >= k else set()
        if not S1 and not S2:
            sim = 1.0
        else:
            sim = len(S1 & S2) / len(S1 | S2) if (S1 or S2) else 0.0
        return float(sim)

    elif method == "dice_kmer":
        k = max(1, int(kwargs.get("k", 3)))
        C1 = Counter(s1[i:i+k] for i in range(len(s1) - k + 1)) if len(s1) >= k else Counter()
        C2 = Counter(s2[i:i+k] for i in range(len(s2) - k + 1)) if len(s2) >= k else Counter()
        inter = sum((C1 & C2).values())
        total = sum(C1.values()) + sum(C2.values())
        sim = (2 * inter) / total if total > 0 else 1.0
        return float(sim)

    elif method == "cosine_kmer":
        k = max(1, int(kwargs.get("k", 3)))
        C1 = Counter(s1[i:i+k] for i in range(len(s1) - k + 1)) if len(s1) >= k else Counter()
        C2 = Counter(s2[i:i+k] for i in range(len(s2) - k + 1)) if len(s2) >= k else Counter()
        sim = _cosine_from_counter(C1, C2)
        return float(sim)

    else:
        raise ValueError(f"Unsupported method: {method}")



def fasta_similarity(
    s1: List[str],
    s2: List[str],
    method: str = "levenshtein",
    processes: Optional[int] = None,
    show_progress: bool = False,
    **kwargs
) -> np.ndarray:
    """Compute pairwise sequence similarity for all (i, j) pairs.

    Args:
        s1: List of query sequences.
        s2: List of target sequences.
        method: Similarity method name.
        processes: Number of worker processes (defaults to cpu_count - 1).
        show_progress: Whether to show a tqdm progress bar.
        **kwargs: Extra arguments forwarded to the selected method.

    Returns:
        np.ndarray with shape (len(s1), len(s2)).
    """
    n1, n2 = len(s1), len(s2)
    if not s1 or not s2:
        return np.zeros((n1, n2), dtype=np.float32)
    index_pairs = [(i, j) for i in range(len(s1)) for j in range(len(s2))]
    nprocs = processes if processes is not None else max(1, cpu_count() - 1)
    chunksize = max(1, len(index_pairs) // (nprocs * 4))
    mat = np.zeros((n1, n2), dtype=np.float32)

    if nprocs == 1:
        _init_worker(s1, s2, method, kwargs)
        iterator = map(_worker, index_pairs)
        pairs = (
            list(tqdm(iterator, total=len(index_pairs)))
            if show_progress
            else list(iterator)
        )
    else:
        with Pool(
            nprocs,
            initializer=_init_worker,
            initargs=(s1, s2, method, kwargs),
            maxtasksperchild=1000,
        ) as pool:
            it = pool.imap(_worker, index_pairs, chunksize=chunksize)
            pairs = list(tqdm(it, total=len(index_pairs))) if show_progress else list(it)
    for i, j, sim in pairs:
        mat[i, j] = float(sim)
    np.nan_to_num(mat, copy=False)
    return mat



if __name__ == "__main__":
    seqs1 = ["ALAGGGPCR", "PEPTIDE"]
    seqs2 = ["ALAGGGPCQ", "PEPTIDER"]

    print("=== Single-pair via batch interface ===")
    res = fasta_similarity(["ALAGGGPCR"], ["ALAGGGPCQ"], method="levenshtein")
    print(f"levenshtein: {float(res[0, 0]):.3f}")

    print("\n=== Batch Similarity Tests ===")
    for m in ["levenshtein", "jaccard_kmer", "nw_blosum62"]:
        print(f"\n--- {m} ---")
        mat = fasta_similarity(seqs1, seqs2, method=m, show_progress=False)
        print(mat)
