"""
eda.py

Team member: Ehsan
Task: Exploratory Data Analysis (EDA)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def run_eda(df, features, label_col='SepsisLabel', bins=30, figsize=(8, 6)):
    """
    For each feature in `features`, plot a figure with two vertically stacked
    probability distributions:
      - Top:   df[feat] for rows where label_col == 0
      - Bottom: df[feat] for rows where label_col == 1
    """
    for feat in features:
        if feat not in df.columns:
            print(f"  • skipping '{feat}' (not in DataFrame)")
            continue

        data0 = df[df[label_col] == 0][feat].dropna()
        data1 = df[df[label_col] == 1][feat].dropna()

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        sns.histplot(data0, bins=bins, stat='density', element='step', ax=axes[0])
        axes[0].set_title(f"{feat} distribution (label = 0)")
        axes[0].set_ylabel("Density")

        sns.histplot(data1, bins=bins, stat='density', element='step', ax=axes[1], color='C1')
        axes[1].set_title(f"{feat} distribution (label = 1)")
        axes[1].set_xlabel(feat)
        axes[1].set_ylabel("Density")

        plt.tight_layout()
        plt.show()


def run_comprehensive_eda(
    df,
    features,
    label_col='SepsisLabel',
    steps=None,
    bins=30,
    figsize_map=(10, 4),
    figsize_corr=(6, 5),
    figsize_box=(6, 4),
    figsize_kde=(12, 3),
    kde_cols=4,
    figsize_pca=(6, 6),
    min_count=2
):
    """
    Rich EDA with selectable steps:
      1) Missingness bar chart
      2) Correlation by label (pairwise deletion, same scale)
      3) Boxplots by label
      4) KDE overlays
      5) PCA scatter

    min_count: minimum non-null values a feature must have in that label group
    to be included in its correlation matrix.
    """
    if steps is None:
        steps = [1, 2, 3, 4, 5]

    # 1) Missingness
    if 1 in steps:
        miss_pct = df[features].isnull().mean().sort_values(ascending=False)
        plt.figure(figsize=figsize_map)
        sns.barplot(x=miss_pct.index, y=miss_pct.values)
        plt.xticks(rotation=90)
        plt.ylabel("Fraction missing")
        plt.title("Missing data percentage per feature")
        plt.tight_layout()
        plt.show()

    # 2) Correlation by label with pairwise deletion
    if 2 in steps:
        corr_dict = {}
        for lbl in [0, 1]:
            subset = df[df[label_col] == lbl]
            counts = subset[features].notnull().sum()
            good = counts[counts >= min_count].index.tolist()
            if len(good) < 2:
                print(f"  • label={lbl}: only {len(good)} features ≥{min_count} non-null → skip")
                corr_dict[lbl] = None
            else:
                # pairwise deletion: each corr(i,j) uses all rows where both i and j are non-null
                corr_dict[lbl] = subset[good].corr()

        # determine common vmin/vmax across both matrices
        mats = [m for m in corr_dict.values() if m is not None]
        if mats:
            vmin = min(m.values.min() for m in mats)
            vmax = max(m.values.max() for m in mats)
        else:
            vmin, vmax = -1, 1

        # plot each
        for lbl, corr in corr_dict.items():
            if corr is None:
                continue
            plt.figure(figsize=figsize_corr)
            sns.heatmap(
                corr,
                cmap="RdBu_r",
                center=0,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'Pearson r'}
            )
            plt.title(f"Correlation Matrix (label = {lbl})")
            plt.tight_layout()
            plt.show()

    # 3) Boxplots
    if 3 in steps:
        for feat in features:
            if feat not in df.columns:
                continue
            plt.figure(figsize=figsize_box)
            sns.boxplot(x=label_col, y=feat, data=df, palette="Set2")
            plt.title(f"{feat} by {label_col}")
            plt.tight_layout()
            plt.show()

    # 4) KDE overlays
    if 4 in steps:
        n = len(features)
        n_rows = int(np.ceil(n / kde_cols))
        fig, axes = plt.subplots(n_rows, kde_cols,
                                 figsize=(figsize_kde[0], figsize_kde[1] * n_rows))
        axes = axes.flatten()
        for ax, feat in zip(axes, features):
            if feat not in df.columns:
                ax.axis('off')
                continue
            sns.kdeplot(
                data=df[df[label_col] == 0][feat].dropna(),
                ax=ax, label=f"{label_col}=0", color="C0"
            )
            sns.kdeplot(
                data=df[df[label_col] == 1][feat].dropna(),
                ax=ax, label=f"{label_col}=1", color="C1"
            )
            ax.set_title(feat)
            ax.legend()
        for ax in axes[n:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 5) PCA scatter
    if 5 in steps:
        sub = df[features + [label_col]].dropna()
        X = (sub[features] - sub[features].mean()) / sub[features].std()
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        pc_df[label_col] = sub[label_col].values

        plt.figure(figsize=figsize_pca)
        sns.scatterplot(
            x='PC1', y='PC2', hue=label_col,
            data=pc_df, alpha=0.6, palette={0: 'C0', 1: 'C1'}
        )
        plt.title(f"PCA of {len(features)} features")
        plt.tight_layout()
        plt.show()




def corr_difference_analysis(
    df,
    features,
    label_col='SepsisLabel',
    min_count=2,
    top_k=10,
    figsize=(6,5)
):
    """
    1) Compute corr0 and corr1 on features with >=min_count non‐null per label
    2) diff = corr1 – corr0
    3) Plot heatmap of diff on [-1,1] scale
    4) Return top_k pairs by absolute change
    """
    # build corr matrices
    corr_mats = {}
    for lbl in [0,1]:
        sub = df[df[label_col]==lbl]
        counts = sub[features].notnull().sum()
        good = counts[counts>=min_count].index.tolist()
        if len(good)<2:
            raise ValueError(f"Not enough features for label={lbl}")
        corr_mats[lbl] = sub[good].corr()

    corr0, corr1 = corr_mats[0], corr_mats[1]
    # align indices & columns (in case they differ)
    common = corr0.index.intersection(corr1.index)
    corr0 = corr0.loc[common, common]
    corr1 = corr1.loc[common, common]

    diff = corr1 - corr0

    # 3) plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        diff, 
        cmap="coolwarm", 
        center=0, 
        vmin=-1, vmax=1, 
        cbar_kws={'label':'Δ Pearson r (1−0)'}
    )
    plt.title("Correlation Difference (label=1 minus label=0)")
    plt.tight_layout()
    plt.show()

    # 4) find top_k absolute changes
    absdiff = diff.abs()
    # only upper triangle, no diagonals
    pairs = []
    for i, fi in enumerate(common):
        for fj in common[i+1:]:
            pairs.append((fi, fj, float(absdiff.at[fi,fj]), float(diff.at[fi,fj])))
    # sort by abs change desc
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:top_k]

    print(f"Top {top_k} feature‐pairs by |Δcorr|:")
    print(" Feature1    Feature2    Δcorr    abs(Δcorr)")
    for f1, f2, ad, d in top:
        print(f" {f1:10s}  {f2:10s}  {d:+.3f}    {ad:.3f}")

    return diff, top
