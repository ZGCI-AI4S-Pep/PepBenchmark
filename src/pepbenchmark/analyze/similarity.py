# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

from tqdm import tqdm


def sliding_window_aar(pair: Tuple[str, str]) -> float:
    seq1, seq2 = pair
    max_match = 0
    len1, len2 = len(seq1), len(seq2)
    for offset in range(-len2 + 1, len1):
        matches = 0
        for i in range(len1):
            j = i - offset
            if 0 <= j < len2:
                if seq1[i] == seq2[j]:
                    matches += 1
        max_match = max(max_match, matches)
    return max_match / max(len1, len2)


# --------------------------------
# 计算数据集内序列两两相似度
# --------------------------------


def _init_worker(_samples, _smi_func):
    global shared_samples, shared_smi_function
    shared_samples = _samples
    shared_smi_function = _smi_func


def _wrapper(pair: Tuple[int, int]) -> Tuple[Tuple[int, int], float | None]:
    i, j = pair
    try:
        return ((i, j), shared_smi_function((shared_samples[i], shared_samples[j])))
    except Exception:
        return ((i, j), None)


def calculate_similarity_each_pair(
    samples: List[str], smi_function=sliding_window_aar, processes: Optional[int] = None
) -> List[Tuple[Tuple[int, int], float]]:
    n = len(samples)
    if processes is None:
        processes = max(1, cpu_count() - 1)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    with Pool(
        processes, initializer=_init_worker, initargs=(samples, smi_function)
    ) as pool:
        results = list(tqdm(pool.imap(_wrapper, index_pairs), total=len(index_pairs)))
    # 只返回sim不为None的pair
    return [(ij, sim) for (ij, sim) in results if sim is not None]


def get_max_similarity_per_sample(
    sequences: List[str],
    processes: Optional[int] = None,
    smi_function=sliding_window_aar,
) -> List[float]:
    # 保证processes为int类型
    n_cpu = max(1, cpu_count() - 1)
    proc = processes if processes is not None else n_cpu
    pair_similarities = calculate_similarity_each_pair(sequences, smi_function, proc)
    n = len(sequences)
    max_sim_per_seq = [0.0] * n
    for (i, j), sim in pair_similarities:
        max_sim_per_seq[i] = max(max_sim_per_seq[i], sim)
        max_sim_per_seq[j] = max(max_sim_per_seq[j], sim)
    return max_sim_per_seq


# --------------------------------
# 计算两数据集间序列两两相似度
# --------------------------------
def _init_worker_cross_dataset(_samples1, _samples2, _smi_func):
    """初始化跨数据集相似度计算的worker函数"""
    global shared_samples1, shared_samples2, shared_smi_function
    shared_samples1 = _samples1
    shared_samples2 = _samples2
    shared_smi_function = _smi_func


def _wrapper_cross_dataset(
    pair: Tuple[int, int],
) -> Tuple[Tuple[int, int], float | None]:
    """跨数据集相似度计算的包装函数"""
    i, j = pair
    try:
        return ((i, j), shared_smi_function((shared_samples1[i], shared_samples2[j])))
    except Exception:
        return ((i, j), None)


def calculate_cross_dataset_similarity(
    samples1: List[str],
    samples2: List[str],
    smi_function=sliding_window_aar,
    processes: Optional[int] = None,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    计算两个数据集之间所有序列对之间的相似度。

    Args:
        samples1: 第一个数据集的序列列表
        samples2: 第二个数据集的序列列表
        smi_function: 相似度计算函数，默认为sliding_window_aar
        processes: 并行进程数，默认为CPU核心数-1

    Returns:
        包含所有序列对相似度的列表，格式为[((i, j), similarity_score)]
        其中i是samples1中的索引，j是samples2中的索引
    """
    if processes is None:
        processes = max(1, cpu_count() - 1)

    # 生成所有跨数据集的索引对
    index_pairs = [(i, j) for i in range(len(samples1)) for j in range(len(samples2))]

    with Pool(
        processes,
        initializer=_init_worker_cross_dataset,
        initargs=(samples1, samples2, smi_function),
    ) as pool:
        results = list(
            tqdm(pool.imap(_wrapper_cross_dataset, index_pairs), total=len(index_pairs))
        )

    # 只返回sim不为None的pair
    return [(ij, sim) for (ij, sim) in results if sim is not None]


def get_max_cross_dataset_similarity_per_sample(
    samples1: List[str], samples2: List[str], processes: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    计算每个样本在另一个数据集中的最大相似度。

    Args:
        samples1: 第一个数据集的序列列表
        samples2: 第二个数据集的序列列表
        processes: 并行进程数，默认为CPU核心数-1

    Returns:
        包含两个列表的元组：
        - samples1中每个序列在samples2中的最大相似度
        - samples2中每个序列在samples1中的最大相似度
    """
    n_cpu = max(1, cpu_count() - 1)
    proc = processes if processes is not None else n_cpu

    pair_similarities = calculate_cross_dataset_similarity(
        samples1, samples2, sliding_window_aar, proc
    )

    # 初始化最大相似度列表
    max_sim_samples1 = [0.0] * len(
        samples1
    )  # samples1中每个序列在samples2中的最大相似度
    max_sim_samples2 = [0.0] * len(
        samples2
    )  # samples2中每个序列在samples1中的最大相似度

    # 更新最大相似度
    for (i, j), sim in pair_similarities:
        max_sim_samples1[i] = max(max_sim_samples1[i], sim)
        max_sim_samples2[j] = max(max_sim_samples2[j], sim)

    return max_sim_samples1, max_sim_samples2
