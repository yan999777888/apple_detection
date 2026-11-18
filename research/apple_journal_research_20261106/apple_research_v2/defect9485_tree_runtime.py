"""Batch numeric tree traversal across trees and proposals at inference time."""

import numpy as np
from scipy.special import expit, softmax


class PackedTrees:
    def __init__(self, model):
        self.is_boosted = hasattr(model, "_predictors")
        if self.is_boosted:
            trees = [p for round_ in model._predictors for p in round_]
            self.bias = model._baseline_prediction
            self.channels = self.bias.shape[1]
        else:
            trees = [est.tree_ for est in model.estimators_]
        left = []
        right = []
        default_left = []
        features = []
        cutoffs = []
        values = []
        roots = []
        offset = 0
        self.depth = 0
        for tree in trees:
            roots.append(offset)
            if self.is_boosted:
                nodes = tree.nodes
                if nodes["is_categorical"].any():
                    raise ValueError("Categorical trees unsupported")
                leaf = nodes["is_leaf"].astype(bool)
                l, r = nodes["left"], nodes["right"]
                feature, threshold = nodes["feature_idx"], nodes["num_threshold"]
                missing = nodes["missing_go_to_left"]
                val = nodes["value"][:, None]
                depth = nodes["depth"].max()
            else:
                l, r = tree.children_left, tree.children_right
                leaf = l < 0
                feature, threshold = tree.feature, tree.threshold
                missing = tree.missing_go_to_left
                val = tree.value[:, :, 0]
                depth = tree.max_depth
            here = np.arange(len(l))
            left.append(np.where(leaf, here, l).astype(np.int32) + offset)
            right.append(np.where(leaf, here, r).astype(np.int32) + offset)
            features.append(np.where(leaf, 0, feature).astype(np.int32))
            cutoffs.append(threshold)
            default_left.append(missing)
            values.append(val)
            offset += len(l)
            self.depth = max(self.depth, int(depth))
        self.roots = np.array(roots, np.int32)
        self.left = np.concatenate(left)
        self.right = np.concatenate(right)
        self.feature = np.concatenate(features)
        self.cutoff = np.concatenate(cutoffs)
        self.missing = np.concatenate(default_left).astype(bool)
        self.values = np.concatenate(values)

    def _raw(self, x):
        x = np.asarray(x, dtype=np.float64 if self.is_boosted else np.float32)
        n = len(x)
        rows = np.arange(n)[:, None]
        ids = np.broadcast_to(self.roots, (n, len(self.roots)))
        has_nan = np.isnan(x).any()
        for _ in range(self.depth):
            obs = x[rows, self.feature[ids]]
            to_left = obs <= self.cutoff[ids]
            if has_nan:
                to_left = np.where(np.isnan(obs), self.missing[ids], to_left)
            ids = np.where(to_left, self.left[ids], self.right[ids])
        vals = self.values[ids]
        if not self.is_boosted:
            return vals.mean(axis=1)
        return (
            vals[:, :, 0]
            .reshape(n, len(self.roots) // self.channels, self.channels)
            .sum(1)
            + self.bias
        )

    def predict_proba(self, x):
        logits = self._raw(x)
        if self.channels > 1:
            return softmax(logits, axis=1)
        p = expit(logits[:, 0])
        return np.column_stack((1 - p, p))

    def predict(self, x):
        return self._raw(x)


def pack(models):
    return tuple(PackedTrees(m) for m in models)
