"""Unsupervised analysis for the Atmos project: PCA + hierarchical clustering.

Unit of analysis: calendar DAYS described by their pollution profile
(basin-mean PM2.5, O3, NO2, TEMP, WIND). This answers a real analytical
question: do days fall into recognizable multi-pollutant regimes
(e.g. hot stagnant high-ozone days vs cool ventilated clean days)?

Method (documented choices):
* StandardScaler (z-scores) fitted on the FULL analysis matrix - this is
  unsupervised exploratory analysis (no target), so no train/test split
  is required; no labels are used anywhere in fitting.
* Ward linkage on euclidean distances (dendrogram + silhouette).
* k selected by comparing candidate k=2..8 with silhouette scores and
  inspecting the dendrogram - NOT hard-coded.
* Clusters are data-driven patterns in THIS dataset, not objectively
  existing categories of days; interpretation is post-hoc profiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CLUSTER_VARS = ["PM25", "O3", "NO2", "TEMP", "WIND"]


def prepare_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    d = panel.dropna(subset=CLUSTER_VARS).copy()
    X = StandardScaler().fit_transform(d[CLUSTER_VARS].values)
    return X, d


def pca_analysis(X: np.ndarray) -> dict:
    p = PCA().fit(X)
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    Z = p.transform(X)
    # variable loadings on the first components (correlation with PCs)
    loadings = pd.DataFrame(
        p.components_.T * np.sqrt(p.explained_variance_),
        index=CLUSTER_VARS, columns=[f"PC{i+1}" for i in range(len(CLUSTER_VARS))])
    return {
        "model": p,
        "explained_variance_ratio": evr,
        "cumulative_variance": cum,
        "n_components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "scores": Z,
        "loadings": loadings,
    }


def hierarchical_fit(X: np.ndarray, k_range: range = range(2, 9)) -> dict:
    Z = linkage(X, method="ward")
    sil = {}
    for k in k_range:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sil[k] = float(silhouette_score(X, labels))
    best_k = max(sil, key=sil.get)
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    return {"linkage_matrix": Z, "silhouette_by_k": sil, "best_k": best_k,
            "labels": labels}


def profile_clusters(d: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    dd = d.assign(cluster=labels)
    prof = dd.groupby("cluster")[CLUSTER_VARS].mean().round(2)
    prof["n_days"] = dd.groupby("cluster").size()
    if "o3_exceed" in dd.columns:
        prof["o3_exceed_share"] = dd.groupby("cluster")["o3_exceed"].mean().round(3)
    return prof


def run_clustering(panel: pd.DataFrame) -> dict:
    X, d = prepare_matrix(panel)
    pca = pca_analysis(X)
    hier = hierarchical_fit(X)
    profile = profile_clusters(d, hier["labels"])
    return {
        "n_days_used": len(d),
        "pca": pca,
        "hierarchical": hier,
        "cluster_profile": profile,
        "labels": hier["labels"],
        "dates": d["date"],
    }
