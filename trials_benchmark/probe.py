"""Multi-label linear probe."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


@dataclass
class ProbeConfig:
    batch_size: int = 8192
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    weight_decay: float = 0.0
    num_workers: int = 0
    eval_every: int = 25          # epochs between validation checks


class _Dataset(Dataset):
    def __init__(self, X, labels, questions):
        self.X = torch.FloatTensor(X)
        self.y = torch.stack([torch.FloatTensor(labels[q]) if q in labels
                              else torch.zeros(len(X)) for q in questions], dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _pos_weights(labels, questions):
    w = []
    for q in questions:
        y = labels.get(q)
        if y is None or len(y) == 0:
            w.append(1.0)
            continue
        pos, neg = np.sum(y == 1), np.sum(y == 0)
        w.append(1.0 if pos == 0 else neg / pos)
    return torch.FloatTensor(w)


def mean_auc(y_true, probs, questions):
    """Mean AUC over questions that have both classes in this split."""
    aucs = [roc_auc_score(y_true[q], probs[:, i]) for i, q in enumerate(questions)
            if q in y_true and len(np.unique(y_true[q])) > 1]
    return float(np.mean(aucs)) if aucs else float("nan")


class LinearProbe:
    def __init__(self, questions, config=ProbeConfig(), device=None):
        self.questions = list(questions)
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.best_epoch = None
        self.best_val_auc = None

    def fit(self, X, labels, seed=None, val=None):
        """Train the probe. With val=(X_val, labels_val), keep the epoch with the best
        mean validation AUC; without it, keep the last epoch."""
        if seed is not None:
            torch.manual_seed(seed)
        c = self.config
        self.model = nn.Linear(X.shape[1], len(self.questions)).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weights(labels, self.questions).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        loader = DataLoader(_Dataset(X, labels, self.questions), batch_size=c.batch_size,
                            shuffle=True, num_workers=c.num_workers,
                            pin_memory=self.device.type == "cuda")
        best_state = None
        for epoch in range(1, c.num_epochs + 1):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
            if val is not None and (epoch % c.eval_every == 0 or epoch == c.num_epochs):
                auc = mean_auc(val[1], self.predict_proba(val[0]), self.questions)
                if self.best_val_auc is None or auc > self.best_val_auc:
                    self.best_val_auc, self.best_epoch = auc, epoch
                    best_state = copy.deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch = c.num_epochs
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        loader = DataLoader(torch.FloatTensor(X), batch_size=self.config.batch_size, shuffle=False)
        return torch.cat([torch.sigmoid(self.model(xb.to(self.device))).cpu() for xb in loader]).numpy()
