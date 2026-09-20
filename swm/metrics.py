"""Evaluation metrics for the first-phase experiments."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skmetrics


def ber(true_bits, pred_bits) -> float:
    return float(np.mean(np.asarray(true_bits) != np.asarray(pred_bits)))


def bit_word_acc(true_bits, pred_bits):
    true_bits = np.asarray(true_bits)
    pred_bits = np.asarray(pred_bits)
    ba = float(np.mean(true_bits == pred_bits))
    wa = float(np.all(true_bits == pred_bits))
    return ba, wa


def cross_talk_matrix(ell_list, true_bits_list, K_ind):
    """Average normalized matched-filter crosstalk.
    X[i, j] = mean over samples with b_i active of |ell_j| / mean_k |ell_k|.
    High off-diagonal entries indicate inter-sector leakage."""
    L = len(ell_list[0]) if ell_list else K_ind  # capacity mode concatenates channels
    X = np.zeros((L, L))
    cnt = np.zeros(L)
    for ell, tb in zip(ell_list, true_bits_list):
        norm = max(float(np.abs(ell).mean()), 1e-12)
        for i in range(L):
            if tb[i] > 0:
                X[i] += np.abs(ell) / norm
                cnt[i] += 1
    for i in range(L):
        if cnt[i] > 0:
            X[i] /= cnt[i]
    return X


def roc(scores_w, scores_nw):
    """AUC / TPR@1%FPR / acc from watermark-vs-no-watermark scores."""
    labels = [0] * len(scores_nw) + [1] * len(scores_w)
    preds = list(scores_nw) + list(scores_w)
    fpr, tpr, _ = skmetrics.roc_curve(labels, preds, pos_label=1)
    auc = float(skmetrics.auc(fpr, tpr))
    idx = np.where(fpr < 0.01)[0]
    tpr1 = float(tpr[idx[-1]]) if len(idx) else float("nan")
    acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    return auc, tpr1, acc


def carrier_max_corr(C, N):
    """Weighted max off-diagonal |cross-correlation| of the code rows."""
    w = np.sqrt(np.asarray(N, dtype=float))
    G = np.asarray(C) * w[None, :]
    norms = np.linalg.norm(G, axis=1)
    G = G / norms[:, None]
    Gamma = G @ G.T
    np.fill_diagonal(Gamma, 0.0)
    return float(np.abs(Gamma).max()), Gamma
