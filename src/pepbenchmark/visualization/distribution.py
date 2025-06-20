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

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aa_order = list("ACDEFGHIKLMNPQRSTVWY")


def plot_peptide_distribution_spited(data_dict, dataset_name, type):
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = ["Times New Roman"]

    titles, combin_df = data_dict.keys(), data_dict.values()

    # Peptide Length Distribution
    # setting seaborn style
    sns.set(style="whitegrid", font_scale=1.1)
    # setting figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for _, (df, ax, title) in enumerate(zip(combin_df, axes, titles)):
        df1 = df.copy()
        df1["length"] = df1["sequence"].apply(len)
        if type == "binary_classification":
            df1["label"] = df1["label"].map({0: "negative", 1: "positive"})
            sns.kdeplot(
                data=df1,
                x="length",
                hue="label",
                fill=True,
                common_norm=False,
                alpha=0.5,
                ax=ax,
            )
        else:
            sns.kdeplot(
                data=df1, x="length", fill=True, common_norm=False, alpha=0.5, ax=ax
            )
            ax.set_title(title, fontsize=14)

    plt.suptitle(
        f"Peptide Length Distribution in splited {dataset_name} dataset.", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # The frequency of amino acids in splited dataset
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for i, (df, ax, title) in enumerate(zip(combin_df, axes, titles)):
        freq_df = compute_aa_freq(df)
        sns.barplot(
            data=freq_df,
            x="AA",
            y="freq",
            hue="label",
            order=aa_order,
            ax=ax,
            palette="muted",
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Amino acids")
        if i == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.suptitle(" The frequency of amino acids in splited dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_peptide_distribution(df, dataset_name, type):
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = ["Times New Roman"]

    sns.set(style="whitegrid", font_scale=1.1)

    df1 = df.copy()
    df1["length"] = df1["sequence"].apply(len)
    if type == "binary_classification":
        df1["label"] = df1["label"].map({0: "negative", 1: "positive"})
        sns.kdeplot(
            data=df1, x="length", hue="label", fill=True, common_norm=False, alpha=0.5
        )
    else:
        sns.kdeplot(data=df1, x="length", fill=True, common_norm=False, alpha=0.5)

    plt.title(f"Peptide Length Distribution in {dataset_name} dataset", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    freq_df = compute_aa_freq(df)
    sns.barplot(
        data=freq_df, x="AA", y="freq", hue="label", order=aa_order, palette="muted"
    )
    plt.xlabel("Amino acids")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.title(" The frequency of amino acids in the dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    df_kmer = compute_all_kmer_freq(df, k_values=[2, 3, 4, 5, 6, 7])
    plot_kmer_freq_multiple(df_kmer, [2, 3, 4, 5, 6, 7], dataset_name=dataset_name)


# -------------   --------------------------------


def compute_aa_freq(df):
    freq_dict = {0: Counter(), 1: Counter()}
    count_total = {0: 0, 1: 0}

    for _, row in df.iterrows():
        label = row["label"]
        seq = row["sequence"]
        freq_dict[label].update(seq)
        count_total[label] += len(seq)

    freq_df = {}
    for label in [0, 1]:
        total = count_total[label]
        freq_list = []
        for aa in aa_order:
            freq = freq_dict[label][aa] / total if total > 0 else 0
            freq_list.append(
                {
                    "AA": aa,
                    "freq": freq,
                    "label": "positive" if label == 1 else "negative",
                }
            )
        freq_df[label] = pd.DataFrame(freq_list)
    return pd.concat(freq_df.values())


def get_kmer_freq(sequences, k):
    counter = Counter()
    seq_counter = defaultdict(set)
    total = 0
    num_sequences = len(sequences)

    for idx, seq in enumerate(sequences):
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            counter[kmer] += 1
            seq_counter[kmer].add(idx)
            total += 1

    freq = {kmer: v / num_sequences for kmer, v in counter.items()}

    seq_count = {kmer: len(seq_counter[kmer]) for kmer in seq_counter}

    seq_ratio = {kmer: count / num_sequences for kmer, count in seq_count.items()}

    return freq, counter, seq_count, seq_ratio


def compute_all_kmer_freq(df, k_values=None):
    """Compute k-mer frequencies for positive and negative sequences in a DataFrame."""
    k_values = [2, 3, 4, 5, 6, 7, 8, 9] if k_values is None else k_values
    pos_seqs = df[df["label"] == 1]["sequence"].tolist()
    neg_seqs = df[df["label"] == 0]["sequence"].tolist()
    records = []
    for k in k_values:
        pos_freq, pos_count, pos_seq_count, pos_ratio = get_kmer_freq(pos_seqs, k)
        neg_freq, neg_count, neg_seq_count, neg_ratio = get_kmer_freq(neg_seqs, k)
        all_kmers = set(pos_ratio.keys()).union(neg_ratio.keys())
        for kmer in all_kmers:
            records.append(
                {
                    "k": k,
                    "kmer": kmer,
                    "label": "positive",
                    "freq": pos_ratio.get(kmer, 0),
                    "count": pos_seq_count.get(kmer, 0),
                }
            )
            records.append(
                {
                    "k": k,
                    "kmer": kmer,
                    "label": "negative",
                    "freq": neg_ratio.get(kmer, 0),
                    "count": neg_seq_count.get(kmer, 0),
                }
            )
    return pd.DataFrame(records)


def plot_kmer_freq_ax(df, k, ax, dataset_name="Dataset"):
    """ """
    sub_df = df[df["k"] == k].copy()
    pivot = sub_df.pivot(index="kmer", columns="label", values="count").fillna(0)
    pivot["diff"] = (pivot.get("positive", 0) - pivot.get("negative", 0)).abs()
    top_kmers = pivot["diff"].sort_values(ascending=False).head(10).index

    plot_df = sub_df[sub_df["kmer"].isin(top_kmers)].copy()
    plot_df["kmer"] = pd.Categorical(
        plot_df["kmer"], categories=top_kmers, ordered=True
    )

    barplot = sns.barplot(
        data=plot_df, x="kmer", y="count", hue="label", palette="muted", ax=ax
    )
    for bar in barplot.patches:
        height = bar.get_height()
        if height == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2

        # kmer = bar.get_x()
        # label = bar.get_label()

        bar_center = bar.get_x() + bar.get_width() / 2
        for _, row in plot_df.iterrows():
            if (
                abs(
                    plot_df["kmer"].cat.categories.get_loc(row["kmer"])
                    - round(bar_center)
                )
                < 1e-2
            ):
                if abs(row["count"] - height) < 1e-5:
                    count = row["count"]
                    ax.text(x, height, f"{count}", ha="center", va="bottom", fontsize=9)
                    break

    ax.set_title(
        f"Top 10 most discriminative {k}-mers between positive and negative samples in {dataset_name}"
    )
    ax.set_xlabel(f"{k}-mers")
    ax.set_ylabel("Count of pos/neg samples")
    ax.tick_params(axis="x", rotation=45)


def plot_kmer_freq_multiple(df, k_values, dataset_name="Dataset"):
    n = len(k_values)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        plot_kmer_freq_ax(df, k, ax, dataset_name)
    plt.tight_layout()
    plt.show()
