
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


aa_order = list('ACDEFGHIKLMNPQRSTVWY')


def plot_peptide_distribution_spited(data_dict, dataset_name, type):
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.family'] = ['Times New Roman']

    titles, combin_df = data_dict.keys(), data_dict.values()


    #Peptide Length Distribution
    # setting seaborn style
    sns.set(style="whitegrid", font_scale=1.1)
    # setting figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (df, ax, title) in enumerate(zip(combin_df, axes, titles)):
        df1 = df.copy()
        df1['length'] = df1['sequence'].apply(len)
        if type == 'binary_classification':
            df1['label'] = df1['label'].map({0: 'negative', 1: 'positive'})
            sns.kdeplot(data=df1, x='length', hue='label', fill=True, common_norm=False, alpha=0.5, ax=ax)
        else:
            sns.kdeplot(data=df1, x='length', fill=True, common_norm=False, alpha=0.5, ax=ax)
            ax.set_title(title, fontsize=14)

    plt.suptitle(f"Peptide Length Distribution in splited {dataset_name} dataset.", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#The frequency of amino acids in splited dataset
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for i, (df, ax, title) in enumerate(zip(combin_df, axes, titles)):
        freq_df = compute_aa_freq(df)
        sns.barplot(data=freq_df, x='AA', y='freq', hue='label', order=aa_order, ax=ax, palette='muted')
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
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.family'] = ['Times New Roman']


    sns.set(style="whitegrid", font_scale=1.1)

    df1 = df.copy()
    df1['length'] = df1['sequence'].apply(len)
    if type == 'binary_classification':
        df1['label'] = df1['label'].map({0: 'negative', 1: 'positive'})
        sns.kdeplot(data=df1, x='length', hue='label', fill=True, common_norm=False, alpha=0.5)
    else:
        sns.kdeplot(data=df1, x='length', fill=True, common_norm=False, alpha=0.5)

    plt.title(f'Peptide Length Distribution in {dataset_name} dataset', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


    freq_df = compute_aa_freq(df)
    sns.barplot(data=freq_df, x='AA', y='freq', hue='label', order=aa_order, palette='muted')
    plt.xlabel("Amino acids")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.title(" The frequency of amino acids in the dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    df_kmer = compute_all_kmer_freq(df, k_values=[2, 3, 4, 5, 6])
    plot_kmer_freq_multiple(df_kmer, [2,3,4,5,6], dataset_name=dataset_name)





#-------------   --------------------------------





def compute_aa_freq(df):
    freq_dict = {0: Counter(), 1: Counter()}
    count_total = {0: 0, 1: 0}

    for _, row in df.iterrows():
        label = row['label']
        seq = row['sequence']
        freq_dict[label].update(seq)
        count_total[label] += len(seq)

    freq_df = {}
    for label in [0, 1]:
        total = count_total[label]
        freq_list = []
        for aa in aa_order:
            freq = freq_dict[label][aa] / total if total > 0 else 0
            freq_list.append({'AA': aa, 'freq': freq, 'label': 'positive' if label == 1 else 'negative'})
        freq_df[label] = pd.DataFrame(freq_list)
    return pd.concat(freq_df.values())


def get_kmer_freq(sequences, k):
    counter = Counter()
    total = 0
    for seq in sequences:
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            counter[kmer] += 1
            total += 1
    # calculate frequency
    freq = {kmer: v/len(sequences) for kmer, v in counter.items()}
    return freq

def compute_all_kmer_freq(df, k_values=[2,3,4,5,6]):
    pos_seqs = df[df['label'] == 1]['sequence'].tolist()
    neg_seqs = df[df['label'] == 0]['sequence'].tolist()
    records = []
    for k in k_values:
        pos_freq = get_kmer_freq(pos_seqs, k)
        neg_freq = get_kmer_freq(neg_seqs, k)
        all_kmers = set(pos_freq.keys()).union(neg_freq.keys())
        for kmer in all_kmers:
            records.append({
                'k': k,
                'kmer': kmer,
                'label': 'positive',
                'freq': pos_freq.get(kmer, 0)
            })
            records.append({
                'k': k,
                'kmer': kmer,
                'label': 'negative',
                'freq': neg_freq.get(kmer, 0)
            })
    return pd.DataFrame(records)


def plot_kmer_freq_ax(df, k, ax, dataset_name="Dataset"):
    """

    """
    sub_df = df[df['k'] == k].copy()
    pivot = sub_df.pivot(index='kmer', columns='label', values='freq').fillna(0)
    pivot['diff'] = (pivot.get('positive', 0) - pivot.get('negative', 0)).abs()
    top_kmers = pivot['diff'].sort_values(ascending=False).head(10).index

    plot_df = sub_df[sub_df['kmer'].isin(top_kmers)]

    sns.barplot(data=plot_df, x='kmer', y='freq', hue='label', palette='muted', ax=ax)
    ax.set_title(f"Top 10 {k}-mers in {dataset_name}")
    ax.set_xlabel(f"{k}-mers")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)


def plot_kmer_freq_multiple(df, k_values, dataset_name="Dataset"):
    n = len(k_values)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        plot_kmer_freq_ax(df, k, ax, dataset_name)
    plt.tight_layout()
    plt.show()

