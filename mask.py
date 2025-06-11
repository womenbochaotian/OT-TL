import numpy as np
import torch
from scipy import optimize

def MAR_mask(X, p=0.3, p_obs=0.5):
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X).float()
    n, d = X.shape
    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d), dtype=bool)
    d_obs = max(int(p_obs * d), 1)
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
    coeffs = _pick_coeffs(X[:, idxs_obs], len(idxs_nas))
    intercepts = _fit_intercepts(X[:, idxs_obs], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_obs] @ coeffs + intercepts)
    ber = torch.rand(n, len(idxs_nas)) if to_torch else np.random.rand(n, len(idxs_nas))
    mask[:, idxs_nas] = ber < ps
    return mask if to_torch else mask.numpy()

def MNAR_mask_logistic(X, p=0.3, p_params=0.3, exclude_inputs=True):
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X).float()
    n, d = X.shape
    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d), dtype=bool)
    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)
    coeffs = _pick_coeffs(X[:, idxs_params], len(idxs_nas))
    intercepts = _fit_intercepts(X[:, idxs_params], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_params] @ coeffs + intercepts)
    ber = torch.rand(n, len(idxs_nas)) if to_torch else np.random.rand(n, len(idxs_nas))
    mask[:, idxs_nas] = ber < ps
    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p if to_torch else np.random.rand(n, d_params) < p
    return mask if to_torch else mask.numpy()

def MNAR_mask_quantiles(X, p=0.3, q=0.2, p_params=0.3, cut='both', MCAR=False):
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X).float()
    n, d = X.shape
    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d), dtype=bool)
    d_na = max(int(p_params * d), 1)
    idxs_na = np.random.choice(d, d_na, replace=False)
    X_na = X[:, idxs_na]
    if cut == 'upper':
        quants = _quantile(X_na, 1 - q, dim=0)
        m = X_na >= quants
    elif cut == 'lower':
        quants = _quantile(X_na, q, dim=0)
        m = X_na <= quants
    else:
        u_quants = _quantile(X_na, 1 - q, dim=0)
        l_quants = _quantile(X_na, q, dim=0)
        m = (X_na <= l_quants) | (X_na >= u_quants)
    ber = torch.rand(n, d_na) if to_torch else np.random.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m
    if MCAR:
        mask = mask | (torch.rand(n, d) < p if to_torch else np.random.rand(n, d) < p)
    return mask if to_torch else mask.numpy()

def _pick_coeffs(X_obs, d_na):
    d_obs = X_obs.shape[1]
    coeffs = torch.randn(d_obs, d_na)
    Wx = X_obs @ coeffs
    coeffs /= torch.std(Wx, dim=0, keepdim=True)
    return coeffs

def _fit_intercepts(X_obs, coeffs, target_p):
    d_na = coeffs.shape[1]
    intercepts = torch.zeros(d_na)
    for j in range(d_na):
        def loss_fn(intercept):
            return torch.sigmoid(X_obs @ coeffs[:, j] + intercept).mean().item() - target_p
        intercepts[j] = optimize.bisect(loss_fn, -50, 50)
    return intercepts

def _quantile(X, q, dim=0):
    if torch.is_tensor(X):
        return X.kthvalue(int(q * X.shape[dim]), dim=dim)[0]
    else:
        return np.quantile(X, q, axis=dim)
