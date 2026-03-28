"""
Top-k motif extraction module for peptide sequence analysis.
"""

import math
import numpy as np
from collections import Counter, defaultdict, OrderedDict
from sklearn.metrics import mutual_info_score, roc_auc_score
from scipy.stats import fisher_exact, chi2_contingency


# ================== 1. Enrichment-based motif extraction ==================
import math
import random
from collections import Counter, defaultdict, OrderedDict

from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import mutual_info_score


import math
from collections import Counter, defaultdict, OrderedDict

from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import mutual_info_score


from collections import Counter, OrderedDict

from collections import Counter, defaultdict, OrderedDict

from collections import Counter, defaultdict, OrderedDict

def extract_motifs_frequency(
    sequences, 
    ks=3, 
    topM=None, 
    top_fraction=None,
    count_mode="presence", 
    min_count=1
):
    """
    Count the most frequent k-mer motifs in sequences, ignoring labels.

    Args:
        sequences: list[str] - Sequence list.
        ks: int or list[int] - k value(s), supports one integer or a list.
        topM: int or None - Return the top N motifs; None keeps all.
        top_fraction: float or None - Return the top fraction of motifs (0~1), higher priority than topM.
        count_mode: str - "presence" | "count"
            - presence: Count each motif once per sequence.
            - count: Count motifs by true occurrence count.
        min_count: int - Minimum motif count.

    Returns:
        dict: {k: OrderedDict({motif: {...}})} or OrderedDict (for a single k)
              Each motif maps to {"count": occurrence count, "freq": relative frequency, "seq_ids": [sequence indices containing it]}
    """
    single_k = isinstance(ks, int)
    if single_k:
        ks = [ks]

    results = {}
    n_seq = len(sequences)

    for k in ks:
        counter = Counter()
        motif2seqs = defaultdict(set)

        # --- Count ---
        for sid, seq in enumerate(sequences):
            if len(seq) < k:
                continue
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            if count_mode == "presence":
                kmers = set(kmers)
            for m in kmers:
                counter[m] += 1
                motif2seqs[m].add(sid)

        motifs_sorted = counter.most_common()

        # --- Filter ---
        filtered = [(m, cnt) for m, cnt in motifs_sorted if cnt >= min_count]

        # --- Truncate ---
        if top_fraction is not None:
            limit = max(1, int(len(filtered) * top_fraction))
            filtered = filtered[:limit]
        elif topM is not None:
            filtered = filtered[:topM]
        # If topM=None and top_fraction=None, keep all motifs.

        # --- Collect results ---
        od = OrderedDict()
        for m, cnt in filtered:
            od[m] = {
                "count": cnt,
                "freq": cnt / n_seq,
                "seq_ids": sorted(list(motif2seqs[m]))
            }

        results[k] = od

    return results[ks[0]] if single_k else results


import math
from collections import Counter, defaultdict, OrderedDict
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import mutual_info_score
import numpy as np


# --------------------------
# Helper: FDR correction
# --------------------------
def fdr_bh(pvals):
    """
    Benjamini-Hochberg FDR correction
    Input: pvals (list of float)
    Output: qvals (list of float), aligned with pvals
    """
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]
    qvals = ranked_pvals * n / (np.arange(n) + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.minimum(qvals, 1.0)
    qvals_final = np.empty_like(qvals)
    qvals_final[order] = qvals
    return qvals_final.tolist()


# --------------------------
# Step 1: Count k-mers
# --------------------------
def count_kmers(sequences, labels, k, count_mode="presence"):
    pos_counter, neg_counter = Counter(), Counter()
    motif2seqs = defaultdict(list)

    for sid, seq in enumerate(sequences):
        if len(seq) < k:
            continue
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        
        if count_mode == "presence":
            # Count each k-mer once per sequence.
            kmers = set(kmers)
            for m in kmers:
                motif2seqs[m].append(sid)
                if labels[sid] == 1:
                    pos_counter[m] += 1
                else:
                    neg_counter[m] += 1
        else:  # count_mode == "count"
            # Count each k-mer by its true number of occurrences.
            # motif2seqs still records each sequence ID only once.
            kmer_counts = Counter(kmers)
            for m, count in kmer_counts.items():
                if sid not in motif2seqs[m]:  # Avoid duplicate sequence IDs.
                    motif2seqs[m].append(sid)
                if labels[sid] == 1:
                    pos_counter[m] += count
                else:
                    neg_counter[m] += count
    return pos_counter, neg_counter, motif2seqs


# --------------------------
# Step 2: Compute motif statistics
# --------------------------
def compute_motif_stats(pos_counter, neg_counter, total_pos, total_neg,
                        motif2seqs, method="fisher", alternative="greater",
                        min_pos_count=3, min_neg_count=3,
                        min_cluster_size=2, strict=False, mode="pos"):
    motif_stats = []
    for m in set(pos_counter.keys()) | set(neg_counter.keys()):
        a = pos_counter[m]
        c = neg_counter[m]
        if a + c < min_cluster_size:
            continue

        if method == "fisher":
            # Ensure contingency table values are never negative.
            # This can happen in count_mode="count" when repeated identical k-mers in one sequence are counted multiple times.
            a = min(a, total_pos)  # Motif count in positives should not exceed total positives.
            c = min(c, total_neg)  # Motif count in negatives should not exceed total negatives.
            
            b = total_pos - a
            d = total_neg - c
            
            # Double-check that all values are non-negative.
            if b < 0 or d < 0:
                # If negatives still exist, skip Fisher test for this motif.
                continue
                
            table = [[a, b], [c, d]]
            try:
                odds_ratio, pval = fisher_exact(table, alternative=alternative)
                motif_stats.append((m, pval, odds_ratio, a, c, "fisher"))
            except Exception as e:
                # If Fisher test fails, log a warning and skip it.
                print(f"Warning: Fisher exact test failed for motif {m}: {e}")
                continue

        elif method == "chi2":
            # Ensure contingency table values are never negative.
            a = min(a, total_pos)
            c = min(c, total_neg)
            
            b = total_pos - a
            d = total_neg - c
            
            # Check that all values are non-negative.
            if b < 0 or d < 0:
                continue
                
            table = [[a, b], [c, d]]
            try:
                chi2, pval, _, _ = chi2_contingency(table, correction=False)
                motif_stats.append((m, pval, chi2, a, c, "chi2"))
            except Exception as e:
                print(f"Warning: Chi2 test failed for motif {m}: {e}")
                continue

        elif method == "ratio":
            pos_ratio = a / total_pos if total_pos > 0 else 0
            neg_ratio = c / total_neg if total_neg > 0 else 0
            score = (pos_ratio + 1e-6) / (neg_ratio + 1e-6)
            motif_stats.append((m, None, score, a, c, "ratio"))

        elif method == "exclusive":
            if strict:
                if c == 0 and a >= min_pos_count and mode in ("pos", "both"):
                    motif_stats.append((m, None, a, a, c, "exclusive_pos"))
                if a == 0 and c >= min_neg_count and mode in ("neg", "both"):
                    motif_stats.append((m, None, c, a, c, "exclusive_neg"))
            else:
                if a >= min_pos_count and c <= min_neg_count and mode in ("pos", "both"):
                    motif_stats.append((m, None, a, a, c, "exclusive_pos"))
                if c >= min_neg_count and a <= min_pos_count and mode in ("neg", "both"):
                    motif_stats.append((m, None, c, a, c, "exclusive_neg"))

        elif method == "logodds":
            pos_ratio = (a + 1) / (total_pos + 2)
            neg_ratio = (c + 1) / (total_neg + 2)
            logodds = math.log(pos_ratio / neg_ratio)
            motif_stats.append((m, None, logodds, a, c, "logodds"))

        elif method == "mutual_info":
            X = [1 if sid in motif2seqs[m] else 0 for sid in range(total_pos + total_neg)]
            Y = [1 if i < total_pos else 0 for i in range(total_pos + total_neg)]
            mi = mutual_info_score(X, Y)
            motif_stats.append((m, None, mi, a, c, "mutual_info"))

    return motif_stats


# --------------------------
# Step 3: Apply FDR correction
# --------------------------
def apply_fdr(motif_stats):
    qval_dict = {}
    raw_pvals = [p for (_, p, _, _, _, _) in motif_stats if p is not None]
    qvals = fdr_bh(raw_pvals)
    for (motif, pval, _, _, _, _), q in zip(
        [ms for ms in motif_stats if ms[1] is not None], qvals
    ):
        qval_dict[motif] = q
    return qval_dict


# --------------------------
# Step 4: Filter + truncate
# --------------------------
def filter_and_truncate(motif_stats, motif2seqs, qval_dict,
                        pval_threshold=None, min_score=None,
                        min_pos=None, min_neg=None,
                        filter_fn=None,
                        topM=50, top_fraction=None,
                        fdr_correct=False):
    # Sort
    def sort_key(x):
        m, pval, score, a, c, method = x
        qval = qval_dict.get(m)
        if method in ("fisher", "chi2"):
            return (qval if fdr_correct else pval, -score)
        elif method in ("ratio", "logodds", "mutual_info"):
            return (-score,)
        elif method.startswith("exclusive"):
            return (-score,)
        return (pval, -score)

    motif_stats.sort(key=sort_key)

    # Filter
    filtered = []
    for m, pval, score, a, c, method in motif_stats:
        qval = qval_dict.get(m)
        keep = True
        if pval_threshold is not None and (pval is None or pval > pval_threshold):
            keep = False
        if min_score is not None and (score is None or score < min_score):
            keep = False
        if min_pos is not None and a < min_pos:
            keep = False
        if min_neg is not None and c < min_neg:
            keep = False
        if filter_fn is not None and not filter_fn(m, pval, score, a, c, method):
            keep = False
        if keep:
            filtered.append((m, pval, score, a, c, method, qval))

    # Truncate
    truncated = filtered
    if top_fraction is not None:
        limit = max(1, int(len(filtered) * top_fraction))
        truncated = filtered[:limit]
    elif topM is not None:
        truncated = filtered[:topM]

    # Convert to OrderedDict
    od = OrderedDict()
    for m, pval, score, a, c, method, qval in truncated:
        od[m] = {
            "score": score,
            "pval": pval,
            "qval": qval,
            "pos_count": a,
            "neg_count": c,
            "method": method,
            "seq_ids": motif2seqs[m],
        }
    return od


# --------------------------
# Main function: dispatch
# --------------------------
def extract_motifs_enriched(
    sequences, labels, ks=5, 
    topM=None, top_fraction=None, 
    mode="pos", count_mode="presence", 
    test_method="fisher", alternative="greater", 
    min_pos_count=3, min_neg_count=3, min_cluster_size=2, strict=False,
    pval_threshold=None, min_score=None, min_pos=None, min_neg=None,
    filter_fn=None, fdr_correct=False
):
    single_k = isinstance(ks, int)
    if single_k:
        ks = [ks]

    results = {}
    pos_ids = [i for i, l in enumerate(labels) if l == 1]
    neg_ids = [i for i, l in enumerate(labels) if l == 0]
    total_pos, total_neg = len(pos_ids), len(neg_ids)

    for k in ks:
        pos_counter, neg_counter, motif2seqs = count_kmers(sequences, labels, k, count_mode)
        motif_stats = compute_motif_stats(
            pos_counter, neg_counter, total_pos, total_neg, motif2seqs,
            method=test_method, alternative=alternative,
            min_pos_count=min_pos_count, min_neg_count=min_neg_count,
            min_cluster_size=min_cluster_size, strict=strict, mode=mode
        )

        qval_dict = apply_fdr(motif_stats) if fdr_correct else {}
        od = filter_and_truncate(
            motif_stats, motif2seqs, qval_dict,
            pval_threshold=pval_threshold, min_score=min_score,
            min_pos=min_pos, min_neg=min_neg,
            filter_fn=filter_fn,
            topM=topM, top_fraction=top_fraction,
            fdr_correct=fdr_correct
        )
        if not od:
            print(f"⚠️ Warning: k={k}, no motifs meet criteria")
        results[k] = od

    return results[ks[0]] if single_k else results


def extract_cluster_sizes(enriched_results):
    """
    Extract cluster sizes for each motif from the output of extract_motifs_enriched.
    Input:
        enriched_results: {k: OrderedDict({motif: {...}})} or {motif: {...}}
    Returns:
        {k: {motif: cluster_size}} or {motif: cluster_size}
    """
    # If the input is already the OrderedDict for a single k.
    if all(isinstance(v, dict) and 'pos_count' in v for v in enriched_results.values()):
        sizes = {}
        for m, info in enriched_results.items():
            pos = info.get("pos_count", 0)
            neg = info.get("neg_count", 0)
            sizes[m] = pos + neg
        return sizes

    # Standard multi-k case.
    cluster_sizes = {}
    for k, motifs in enriched_results.items():
        sizes = {}
        for m, info in motifs.items():
            if isinstance(info, dict):
                pos = info.get("pos_count", 0)
                neg = info.get("neg_count", 0)
                sizes[m] = pos + neg
        cluster_sizes[k] = sizes
    return cluster_sizes


# ================== 2. Evaluation helpers ==================
def compute_joint_mi(sequences, labels, kmers):
    """Compute joint mutual information for multiple k-mers."""
    X_patterns = [tuple(1 if m in seq else 0 for m in kmers) for seq in sequences]
    X_labels = [hash(pat) for pat in X_patterns]
    return mutual_info_score(X_labels, labels)


# ================== 3. Main function ==================
def check_kmer_summary(sequences, labels, k=5, topM=10,
                       test_method="fisher", mode="pos", exclusive_min_count=3):
    """
    Check the k-mer summary statistics.

    Args:
        sequences: list[str] - Sequence list.
        labels: list[int] - Label list (0/1).
        k: int - k value.
        topM: int - Number of top motifs to return.
        test_method: str - Test method.
        mode: str - "pos" | "neg" | "both"
        exclusive_min_count: int - Minimum count threshold for exclusive mode.
    """
    enriched = extract_motifs_enriched(
        sequences, labels, ks=k, topM=topM, mode=mode,
        test_method=test_method
    )
    
    # If a single-k result is returned, use it directly.
    if isinstance(enriched, OrderedDict):
        motifs_dict = enriched
    else:
        motifs_dict = enriched[k]
    
    cluster_sizes = extract_cluster_sizes(motifs_dict)
    
    top_motifs = list(motifs_dict.keys())
    total_sequences = len(sequences)

    # ---- TopM coverage ----
    covered_idx = set()
    for m in top_motifs:
        covered_idx.update(motifs_dict[m]["seq_ids"])
    covered_sequences = len(covered_idx)

    pos_in_cov = sum(1 for i in covered_idx if labels[i] == 1)
    neg_in_cov = sum(1 for i in covered_idx if labels[i] == 0)

    # ---- Mutual information ----
    mi_scores = []
    for m, info in motifs_dict.items():
        seq_set = set(info["seq_ids"])
        X = [1 if sid in seq_set else 0 for sid in range(len(labels))]
        Y = labels
        mi_scores.append(mutual_info_score(X, Y))

    max_mi = max(mi_scores) if mi_scores else 0
    joint_mi = compute_joint_mi(sequences, labels, top_motifs) if top_motifs else 0

    # ---- Exclusive motifs ----
    exclusive_strong = [(m, info["pos_count"]) 
                        for m, info in motifs_dict.items()
                        if info["pos_count"] >= exclusive_min_count and info["neg_count"] == 0]

    # ---- OR ----
    or_vals = []
    for m, info in motifs_dict.items():
        odds = info["score"] if info["method"] in ("fisher", "ratio", "logodds") else None
        if odds not in [None, 0, np.inf, np.nan]:
            try:
                or_vals.append(abs(np.log(float(odds))))
            except Exception:
                pass
    max_or = max(or_vals) if or_vals else 0

    return {
        "cluster_sizes": cluster_sizes,
        "total_sequences": total_sequences,
        "covered_sequences": covered_sequences,
        "pos_in_covered": pos_in_cov,
        "neg_in_covered": neg_in_cov,
        "max_MI": max_mi,
        "joint_MI": joint_mi,
        "exclusive_strong_count": len(exclusive_strong),
        "exclusive_strong_motifs": exclusive_strong,
        "max_OR": max_or
    }


# ================== 4. Print function ==================
def print_kmer_summary(summary):
    """Print k-mer summary statistics."""
    print("\n=== k-mer dataset analysis summary ===")
    print(f"0. Total sequences: {summary['total_sequences']}")
    print(f"1. Sequences covered by TopM motifs: {summary['covered_sequences']}")
    print(f"   Including: positives {summary['pos_in_covered']}, negatives {summary['neg_in_covered']}")
    print(f"2. Mutual information: max_MI={summary['max_MI']:.3f}, joint_MI={summary['joint_MI']:.3f}")
    print(f"3. Strong exclusive motifs (≥{summary['exclusive_strong_count']} and all positive): {summary['exclusive_strong_count']}")
    if summary['exclusive_strong_motifs']:
        for m, a in summary['exclusive_strong_motifs'][:10]:
            print(f"   - {m} occurs {a} times")
    print(f"4. Maximum odds ratio OR: {summary['max_OR']:.3f}")


# ================== 5. Data leakage detection ==================
def check_kmer_leakage(sequences, labels, k=5, topM=10, min_count=3):
    """
    Detect whether the dataset has k-mer leakage.

    Args:
        sequences: list[str] - Sequence list.
        labels: list[int] - Label list (0/1).
        k: int - k value.
        topM: int - Number of top motifs to return.
        min_count: int - Minimum count threshold.
        
    Returns:
        results: dict containing global metrics and each TopM motif group
    """
    # Extract motifs
    motifs_info = extract_motifs_enriched(sequences, labels, ks=k, topM=100000, count_mode="presence")
    
    # If a single-k result is returned, use it directly.
    if isinstance(motifs_info, OrderedDict):
        motifs_dict = motifs_info
    else:
        motifs_dict = motifs_info[k]

    pos_total = sum(labels)
    neg_total = len(labels) - pos_total

    motif_stats = []
    exclusive_list, mi_list, auc_list, or_list = [], [], [], []

    for m, info in motifs_dict.items():
        a = info["pos_count"]  # Count in positive samples.
        c = info["neg_count"]  # Count in negative samples.
        seq_ids = info["seq_ids"]

        if a + c < min_count:  # Filter low-frequency motifs.
            continue

        # Build the presence vector.
        seq_set = set(seq_ids)
        X = [1 if sid in seq_set else 0 for sid in range(len(labels))]
        Y = labels

        # Mutual Information
        mi = mutual_info_score(X, Y)

        # AUC
        try:
            auc = roc_auc_score(Y, X)
        except Exception:
            auc = 0.5

        # Fisher OR
        b = pos_total - a
        d = neg_total - c
        try:
            odds, _ = fisher_exact([[a, b], [c, d]])
        except Exception:
            odds = np.nan

        motif_stats.append((m, a, c, mi, auc, odds))

        mi_list.append((m, mi, a, c))
        auc_list.append((m, auc, a, c))
        if a == 0 or c == 0:
            exclusive_list.append((m, a, c, max(a, c)))
        if odds not in [0, np.inf, np.nan]:
            or_list.append((m, odds, a, c))

    # ---- TopM extraction ----
    mi_topM = sorted(mi_list, key=lambda x: -x[1])[:topM]
    auc_topM = sorted(auc_list, key=lambda x: -x[1])[:topM]
    excl_topM = sorted(exclusive_list, key=lambda x: -x[3])[:topM]
    or_topM = sorted(or_list, key=lambda x: -abs(np.log(x[1])))[:topM] if or_list else []

    # ---- Global metrics ----
    exclusive_rate = len(exclusive_list) / len(motifs_dict) if motifs_dict else 0
    max_mi = mi_topM[0][1] if mi_topM else 0
    max_auc = auc_topM[0][1] if auc_topM else 0.5
    max_or = max(abs(np.log(o[1])) for o in or_list) if or_list else 0

    leakage_index = max(max_mi, max_auc, exclusive_rate)

    results = {
        "global": {
            "exclusive_rate": exclusive_rate,
            "max_MI": max_mi,
            "max_AUC": max_auc,
            "max_OR": max_or,
            "leakage_index": leakage_index,
        },
        "MI_topM": mi_topM,
        "AUC_topM": auc_topM,
        "Exclusive_topM": excl_topM,
        "OR_topM": or_topM
    }
    return results


def print_leakage_results(results, topM=10):
    """Pretty-print leakage detection results."""
    print("\n=== Global metrics ===")
    for k, v in results["global"].items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    print(f"\n=== TopM={topM} motifs with highest mutual information ===")
    for m, mi, a, c in results["MI_topM"][:topM]:
        print(f"{m} | pos:{a} | neg:{c} | MI:{mi:.3f}")

    print(f"\n=== TopM={topM} motifs with highest AUC ===")
    for m, auc, a, c in results["AUC_topM"][:topM]:
        print(f"{m} | pos:{a} | neg:{c} | AUC:{auc:.3f}")

    print(f"\n=== TopM={topM} Exclusive motifs ===")
    for m, a, c, sup in results["Exclusive_topM"][:topM]:
        print(f"{m} | pos:{a} | neg:{c} | support:{sup}")

    print(f"\n=== TopM={topM} motifs with extreme OR ===")
    for m, odds, a, c in results["OR_topM"][:topM]:
        print(f"{m} | pos:{a} | neg:{c} | OR:{odds:.3f}")


# ================== 6. Unified clustering interface ==================
def create_unified_motif_clusters(
    sequences, 
    labels, 
    k=5, 
    topM=50, 
    test_method="fisher",
    min_support=3,
    min_jaccard=0.5,
    merge_strategy="jaccard",
    conflict_strategy="max_support",
    **kwargs
):
    """
    Create motif clustering results in a unified format consistent with cdhit and mmseqs2.

    Main features:
    1. Cluster based on motifs.
    2. Treat all unclustered sequences as independent clusters.
    3. Return the UnifiedClusterResult-style format used by other clustering methods.

    Args:
        sequences: list[str] - Sequence list.
        labels: list[int] - Label list (0/1).
        k: int - k-mer size.
        topM: int - Number of top motifs to extract.
        test_method: str - Motif extraction method.
        min_support: int - Minimum support for motif merging.
        min_jaccard: float - Minimum Jaccard similarity for motif merging.
        merge_strategy: str - Merge strategy ("jaccard", "overlap", "union").
        conflict_strategy: str - Conflict resolution strategy.
        **kwargs: Other arguments.

    Returns:
        dict: A dictionary with full clustering information in a format consistent with cdhit/mmseqs2
        {
            "cluster_assignments": {cluster_id: [seq_indices]},
            "cluster_representatives": {cluster_id: representative_seq_idx},
            "cluster_metadata": {cluster_id: motif_info},
            "total_sequences": int,
            "total_clusters": int,
            "clustering_method": "motif",
            "parameters": dict,
            "unclustered_count": int
        }
    """
    from collections import defaultdict
    import numpy as np
    
    # Step 1: Extract enriched motifs
    motifs_info = extract_motifs_enriched(
        sequences, labels, ks=k, topM=topM, 
        test_method=test_method, min_pos_count=min_support,
        **kwargs
    )
    
    # Handle the single-k case.
    if isinstance(motifs_info, OrderedDict):
        motifs_dict = motifs_info
    else:
        motifs_dict = motifs_info[k] if k in motifs_info else OrderedDict()
    
    # Step 2: Find motif pairs that should be merged.
    merge_pairs = _find_motif_merge_pairs(
        motifs_dict, min_support, min_jaccard, merge_strategy
    )
    
    # Step 3: Build clusters.
    cluster_map = _build_motif_cluster_map(motifs_dict, merge_pairs)
    
    # Step 4: Resolve conflicts.
    cluster_map = _resolve_motif_conflicts(cluster_map, conflict_strategy)
    
    # Step 5: Create the unified `cluster_assignments` structure.
    cluster_assignments = {}
    cluster_representatives = {}
    cluster_metadata = {}
    
    # Process motif-based clusters.
    for cluster_name, cluster_info in cluster_map.items():
        seq_indices = cluster_info['seq_ids']
        if seq_indices:  # Non-empty cluster.
            cluster_assignments[cluster_name] = seq_indices
            cluster_representatives[cluster_name] = seq_indices[0]  # Use the first sequence as the representative.
            cluster_metadata[cluster_name] = {
                'motifs': cluster_info['motifs'],
                'support': cluster_info['support'],
                'size': len(seq_indices)
            }
    
    # Step 6: Handle unclustered sequences by assigning each to its own cluster.
    clustered_seq_indices = set()
    for seq_indices in cluster_assignments.values():
        clustered_seq_indices.update(seq_indices)
    
    unclustered_indices = [
        i for i in range(len(sequences)) 
        if i not in clustered_seq_indices
    ]
    
    # Create an individual cluster for each unclustered sequence.
    for seq_idx in unclustered_indices:
        individual_cluster_id = f"unclustered_seq_{seq_idx}"
        cluster_assignments[individual_cluster_id] = [seq_idx]
        cluster_representatives[individual_cluster_id] = seq_idx
        cluster_metadata[individual_cluster_id] = {
            'motifs': [],
            'support': 1,
            'size': 1,
            'type': 'individual'
        }
    
    # Step 7: Build the final result.
    result = {
        "cluster_assignments": cluster_assignments,
        "cluster_representatives": cluster_representatives, 
        "cluster_metadata": cluster_metadata,
        "total_sequences": len(sequences),
        "total_clusters": len(cluster_assignments),
        "clustering_method": "motif",
        "parameters": {
            "k": k,
            "topM": topM,
            "test_method": test_method,
            "min_support": min_support,
            "min_jaccard": min_jaccard,
            "merge_strategy": merge_strategy,
            "conflict_strategy": conflict_strategy
        },
        "unclustered_count": len(unclustered_indices),
        "motif_based_clusters": len(cluster_map),
        "individual_clusters": len(unclustered_indices)
    }
    
    return result


def _find_motif_merge_pairs(motifs_dict, min_support, min_jaccard, merge_strategy):
    """Find motif pairs that should be merged."""
    motifs = list(motifs_dict.keys())
    merge_pairs = []
    
    for i in range(len(motifs)):
        for j in range(i + 1, len(motifs)):
            motif1, motif2 = motifs[i], motifs[j]
            
            seq_ids1 = set(motifs_dict[motif1]['seq_ids'])
            seq_ids2 = set(motifs_dict[motif2]['seq_ids'])
            
            # Compute similarity.
            intersection = len(seq_ids1 & seq_ids2)
            if intersection < min_support:
                continue
                
            if merge_strategy == "jaccard":
                union = len(seq_ids1 | seq_ids2)
                similarity = intersection / union if union > 0 else 0
            elif merge_strategy == "overlap":
                min_size = min(len(seq_ids1), len(seq_ids2))
                similarity = intersection / min_size if min_size > 0 else 0
            elif merge_strategy == "union":
                similarity = intersection / min_support if min_support > 0 else 0
            else:
                similarity = intersection / min_support if min_support > 0 else 0
            
            if similarity >= min_jaccard:
                merge_pairs.append((motif1, motif2, similarity))
    
    return merge_pairs


def _build_motif_cluster_map(motifs_dict, merge_pairs):
    """Build the cluster mapping from motifs and merge pairs."""
    # Initialize with one cluster per motif.
    cluster_map = {}
    for motif, info in motifs_dict.items():
        cluster_map[f"motif_cluster_{motif}"] = {
            'motifs': [motif],
            'seq_ids': info['seq_ids'].copy(),
            'support': len(info['seq_ids'])
        }
    
    # Merge clusters according to `merge_pairs`.
    motif_to_cluster = {motif: f"motif_cluster_{motif}" for motif in motifs_dict.keys()}
    
    # Sort by similarity in descending order.
    sorted_pairs = sorted(merge_pairs, key=lambda x: x[2], reverse=True)
    
    for motif1, motif2, similarity in sorted_pairs:
        cluster1_name = motif_to_cluster.get(motif1)
        cluster2_name = motif_to_cluster.get(motif2)
        
        if (cluster1_name and cluster2_name and 
            cluster1_name != cluster2_name and 
            cluster1_name in cluster_map and 
            cluster2_name in cluster_map):
            
            # Merge `cluster2` into `cluster1`.
            cluster1 = cluster_map[cluster1_name]
            cluster2 = cluster_map[cluster2_name]
            
            # Update `cluster1`.
            cluster1['motifs'].extend(cluster2['motifs'])
            cluster1['seq_ids'].extend(cluster2['seq_ids'])
            cluster1['seq_ids'] = list(set(cluster1['seq_ids']))  # Deduplicate sequence IDs.
            cluster1['support'] = len(cluster1['seq_ids'])
            
            # Update motif-to-cluster mapping.
            for motif in cluster2['motifs']:
                motif_to_cluster[motif] = cluster1_name
            
            # Remove `cluster2`.
            del cluster_map[cluster2_name]
    
    return cluster_map


def _resolve_motif_conflicts(cluster_map, strategy="max_support"):
    """Resolve conflicts when a sequence belongs to multiple clusters."""
    from collections import defaultdict
    
    # Find conflicting sequences.
    seq_to_clusters = defaultdict(list)
    for cluster_name, cluster_info in cluster_map.items():
        for seq_id in cluster_info['seq_ids']:
            seq_to_clusters[seq_id].append(cluster_name)
    
    conflicts = {seq_id: clusters for seq_id, clusters in seq_to_clusters.items() 
                if len(clusters) > 1}
    
    if not conflicts:
        return cluster_map  # No conflicts.
    
    # Resolve conflicts.
    for seq_id, cluster_names in conflicts.items():
        if strategy == "max_support":
            # Assign the sequence to the cluster with the highest support.
            best_cluster = max(cluster_names, 
                             key=lambda c: cluster_map[c]['support'])
            
            # Remove the sequence from the other clusters.
            for cluster_name in cluster_names:
                if cluster_name != best_cluster and seq_id in cluster_map[cluster_name]['seq_ids']:
                    cluster_map[cluster_name]['seq_ids'].remove(seq_id)
                    cluster_map[cluster_name]['support'] = len(cluster_map[cluster_name]['seq_ids'])
        
        elif strategy == "first":
            # Keep the sequence in the first cluster.
            best_cluster = cluster_names[0]
            for cluster_name in cluster_names[1:]:
                if seq_id in cluster_map[cluster_name]['seq_ids']:
                    cluster_map[cluster_name]['seq_ids'].remove(seq_id)
                    cluster_map[cluster_name]['support'] = len(cluster_map[cluster_name]['seq_ids'])
    
    # Remove empty clusters.
    empty_clusters = [name for name, info in cluster_map.items() 
                     if len(info['seq_ids']) == 0]
    for name in empty_clusters:
        del cluster_map[name]
    
    return cluster_map


def create_unified_cluster_result_from_motifs(
    sequences,
    labels, 
    k=5,
    topM=50,
    **kwargs
):
    """
    Create motif clustering results in `UnifiedClusterResult` format.

    This function wraps `create_unified_motif_clusters`, and the returned
    format can be used directly by `ClusterBasedSplitter`.

    Args:
        sequences: list[str] - sequence list
        labels: list[int] - label list
        k: int - k-mer size
        topM: int - number of top motifs
        **kwargs: additional clustering parameters

    Returns:
        UnifiedClusterResult: unified clustering result object
    """
    cluster_info = create_unified_motif_clusters(
        sequences, labels, k=k, topM=topM, **kwargs
    )
    
    # Import `UnifiedClusterResult` here to avoid circular imports.
    try:
        from pepbenchmark.cluster.interfaces import UnifiedClusterResult
        
        return UnifiedClusterResult(
            cluster_assignments=cluster_info["cluster_assignments"],
            cluster_representatives=cluster_info["cluster_representatives"],
            cluster_metadata=cluster_info["cluster_metadata"],
            total_sequences=cluster_info["total_sequences"], 
            total_clusters=cluster_info["total_clusters"],
            clustering_method=cluster_info["clustering_method"],
            parameters=cluster_info["parameters"],
            quality_metrics={
                "unclustered_count": cluster_info["unclustered_count"],
                "motif_based_clusters": cluster_info["motif_based_clusters"],
                "individual_clusters": cluster_info["individual_clusters"]
            }
        )
    except ImportError:
        # Return the raw dictionary if `UnifiedClusterResult` cannot be imported.
        return cluster_info


# ================== 7. Usage examples ==================
if __name__ == "__main__":
    import pandas as pd
    
    # Load data.
    df = pd.read_csv("/home/batchcom/assist/PepBenchmark/datasets/our/bbp1/fasta.csv")
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()

    print("=== Testing different detection methods ===")
    
    # Test with a single k value.
    print("\n--- Single k value (k=5) test ---")
    motifs_single = extract_motifs_enriched(sequences, labels, ks=5, topM=10, test_method="fisher")
    print("Return type for a single k value:", type(motifs_single))
    print("Top 3 motifs:")
    for i, (m, info) in enumerate(motifs_single.items()):
        if i >= 3:
            break
        print(f"  {m}: pos={info['pos_count']}, neg={info['neg_count']}, score={info['score']:.3f}")

    # Test with multiple k values.
    print("\n--- Multiple k values [3,4,5] test ---")
    motifs_multi = extract_motifs_enriched(sequences, labels, ks=[3,4,5], topM=5, test_method="fisher")
    print("Return type for multiple k values:", type(motifs_multi))
    for k, motifs in motifs_multi.items():
        print(f"k={k}: found {len(motifs)} motifs")

    # Test different methods.
    test_methods = ["fisher", "chi2", "ratio", "exclusive", "logodds", "mutual_info"]
    
    for method in test_methods:
        print(f"\n--- {method.upper()} method ---")
        try:
            if method == "exclusive":
                motifs = extract_motifs_enriched(sequences, labels, ks=5, topM=5, 
                                               test_method=method, min_pos_count=5)
            else:
                motifs = extract_motifs_enriched(sequences, labels, ks=5, topM=5, 
                                               test_method=method)
            
            for m, info in motifs.items():
                score_str = f"{info['score']:.3f}" if isinstance(info['score'], (int, float)) else str(info['score'])
                print(f"  {m}: pos={info['pos_count']}, neg={info['neg_count']}, "
                      f"{method}={score_str}")
        except Exception as e:
            print(f"  Error: {e}")

    # Dataset summary analysis.
    print("\n=== Dataset summary analysis ===")
    summary = check_kmer_summary(sequences, labels, k=5, topM=20, 
                                test_method="fisher", exclusive_min_count=3)
    print_kmer_summary(summary)

    # Data leakage detection.
    print("\n=== Data leakage detection ===")
    leakage_results = check_kmer_leakage(sequences, labels, k=5, topM=10, min_count=3)
    print_leakage_results(leakage_results, topM=5)
