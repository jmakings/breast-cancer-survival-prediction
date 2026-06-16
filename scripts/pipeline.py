# ============================================================
# SURVIVAL DEEP LEARNING PIPELINE (LEAKAGE-SAFE + CV + TEST)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

import copy

# -----------------------------
# GLOBAL SEED
# -----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ============================================================
# COX LOSS (STABLE)
# ============================================================
def cox_loss(risk_scores, times, events):
    order = torch.argsort(times, descending=True)

    risk_scores = risk_scores[order]
    events = events[order]

    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)

    loss = -torch.sum((risk_scores - log_cumsum) * events)

    n_events = events.sum().clamp_min(1.0)

    return loss / n_events


# ============================================================
# SAFE FEATURE FILTER (GLOBAL)
# ============================================================
def remove_constant_features(X, name="X"):
    var = X.var(axis=0)
    keep = var > 1e-8

    print(f"{name}: dropping {(~keep).sum()} constant features")

    return X[:, keep], keep


# ============================================================
# PREPROCESSOR (LEAKAGE SAFE)
# ============================================================
class SurvivalPreprocessor:
    def __init__(self):
        self.scaler_clin = StandardScaler()
        self.scaler_expr = StandardScaler()
        self.is_fitted = False

    def fit(self, X_clin, X_expr):
        self.scaler_clin.fit(X_clin)
        self.scaler_expr.fit(X_expr)
        self.is_fitted = True

    def transform(self, X_clin, X_expr):
        assert self.is_fitted

        X_clin = self.scaler_clin.transform(X_clin)
        X_expr = self.scaler_expr.transform(X_expr)

        X_clin = np.nan_to_num(X_clin, nan=0.0, posinf=0.0, neginf=0.0)
        X_expr = np.nan_to_num(X_expr, nan=0.0, posinf=0.0, neginf=0.0)

        return X_clin, X_expr


# ============================================================
# SURVIVAL AWARE AUTOENCODER
# ============================================================
class SurvivalAwareAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        self.survival_head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        risk = self.survival_head(z).squeeze(-1)
        return x_recon, z, risk


# ============================================================
# DUAL ENCODER SURVIVAL MODEL
# ============================================================
class DualEncoderSurvivalNet(nn.Module):
    def __init__(self, clin_mut_dim, expr_autoencoder,
                 latent_dim_clin=32, latent_dim_expr=64):

        super().__init__()

        self.clin_mut_enc = nn.Sequential(
            nn.Linear(clin_mut_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim_clin),
            nn.ReLU()
        )

        self.expr_enc = expr_autoencoder

        fusion_dim = latent_dim_clin + latent_dim_expr

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x_clin, x_expr):
        z_clin = self.clin_mut_enc(x_clin)
        z_expr = self.expr_enc(x_expr)

        z = torch.cat([z_clin, z_expr], dim=1)
        risk = self.fusion(z).squeeze(-1)

        return risk


# ============================================================
# AUTOENCODER TRAINING (PER FOLD)
# ============================================================
def train_survival_autoencoder(X_expr, y_time, y_event,
                               latent_dim=64, epochs=200):

    input_dim = X_expr.shape[1]
    ae = SurvivalAwareAutoencoder(input_dim, latent_dim)

    optimizer = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    recon_fn = nn.MSELoss()

    X_expr = X_expr.float()
    y_time = y_time.float()
    y_event = y_event.float()

    best_loss = float("inf")
    best_weights = copy.deepcopy(ae.state_dict())

    for epoch in range(epochs):
        ae.train()
        optimizer.zero_grad()

        recon, z, risk = ae(X_expr)

        recon_loss = recon_fn(recon, X_expr)
        surv_loss = cox_loss(risk, y_time, y_event)

        loss = recon_loss + 0.05 * surv_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = copy.deepcopy(ae.state_dict())

    ae.load_state_dict(best_weights)
    return ae


# ============================================================
# TRAIN DUAL ENCODER
# ============================================================
def train_dual_encoder_survival_net(X_clin, X_expr, y_time, y_event,
                                   expr_autoencoder,
                                   epochs=300):

    model = DualEncoderSurvivalNet(
        clin_mut_dim=X_clin.shape[1],
        expr_autoencoder=expr_autoencoder
    )

    X_clin = X_clin.float()
    X_expr = X_expr.float()
    y_time = y_time.float()
    y_event = y_event.float()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        risk = model(X_clin, X_expr)
        loss = cox_loss(risk, y_time, y_event)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


# ============================================================
# C-INDEX
# ============================================================
def compute_cindex(model, times, events, X_clin, X_expr):
    model.eval()
    with torch.no_grad():
        risk = model(X_clin, X_expr).cpu().numpy()

    return concordance_index(times, -risk, events)


# ============================================================
# FULL PIPELINE
# ============================================================
def run_pipeline(X_clin, X_expr, y_time, y_event):

    # -----------------------------
    # 1. GLOBAL CLEANING
    # -----------------------------
    X_clin, keep1 = remove_constant_features(X_clin, "clinical+mutation")
    X_expr, keep2 = remove_constant_features(X_expr, "expression")

    # -----------------------------
    # 2. HOLDOUT TEST SPLIT
    # -----------------------------
    idx = np.arange(len(y_event))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        stratify=y_event,
        random_state=42
    )

    Xc_train, Xc_test = X_clin[train_idx], X_clin[test_idx]
    Xe_train, Xe_test = X_expr[train_idx], X_expr[test_idx]

    yt_train, yt_test = y_time[train_idx], y_time[test_idx]
    ye_train, ye_test = y_event[train_idx], y_event[test_idx]

    # -----------------------------
    # 3. CV SETUP
    # -----------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []

    for fold, (tr, va) in enumerate(skf.split(Xc_train, ye_train)):

        print(f"\nFOLD {fold+1}")

        Xc_tr, Xc_va = Xc_train[tr], Xc_train[va]
        Xe_tr, Xe_va = Xe_train[tr], Xe_train[va]

        yt_tr, yt_va = yt_train[tr], yt_train[va]
        ye_tr, ye_va = ye_train[tr], ye_train[va]

        # preprocess
        pre = SurvivalPreprocessor()
        pre.fit(Xc_tr, Xe_tr)

        Xc_tr, Xe_tr = pre.transform(Xc_tr, Xe_tr)
        Xc_va, Xe_va = pre.transform(Xc_va, Xe_va)

        # tensors
        Xc_tr = torch.tensor(Xc_tr).float()
        Xe_tr = torch.tensor(Xe_tr).float()
        yt_tr = torch.tensor(yt_tr).float()
        ye_tr = torch.tensor(ye_tr).float()

        Xc_va = torch.tensor(Xc_va).float()
        Xe_va = torch.tensor(Xe_va).float()

        # autoencoder
        ae = train_survival_autoencoder(Xe_tr, yt_tr, ye_tr)

        # model
        model = train_dual_encoder_survival_net(
            Xc_tr, Xe_tr, yt_tr, ye_tr, ae
        )

        cindex = compute_cindex(model, yt_va, ye_va, Xc_va, Xe_va)
        cv_scores.append(cindex)

        print("C-index:", cindex)

    print("\nCV Mean:", np.mean(cv_scores))
    print("CV Std:", np.std(cv_scores))

    # -----------------------------
    # 4. FINAL TRAIN ON FULL TRAIN SET
    # -----------------------------
    pre = SurvivalPreprocessor()
    pre.fit(Xc_train, Xe_train)

    Xc_train, Xe_train = pre.transform(Xc_train, Xe_train)
    Xc_test, Xe_test = pre.transform(Xc_test, Xe_test)

    Xc_train = torch.tensor(Xc_train).float()
    Xe_train = torch.tensor(Xe_train).float()
    yt_train = torch.tensor(yt_train).float()
    ye_train = torch.tensor(ye_train).float()

    Xc_test = torch.tensor(Xc_test).float()
    Xe_test = torch.tensor(Xe_test).float()

    ae = train_survival_autoencoder(Xe_train, yt_train, ye_train)

    final_model = train_dual_encoder_survival_net(
        Xc_train, Xe_train, yt_train, ye_train, ae
    )

    test_cindex = compute_cindex(final_model, yt_test, ye_test,
                                Xc_test, Xe_test)

    print("\n========================")
    print("FINAL TEST C-INDEX:", test_cindex)
    print("========================")

    return final_model, cv_scores, test_cindex