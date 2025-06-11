import numpy as np
import torch
from geomloss import SamplesLoss
from scipy import optimize
from sklearn.utils.extmath import randomized_svd

F32PREC = np.finfo(np.float32).eps

# --- Soft Impute ---
def converged(x_old, x, mask, thresh):
    x_old_na = x_old[mask]
    x_na = x[mask]
    rmse = np.sqrt(np.sum((x_old_na - x_na) ** 2))
    denom = np.sqrt((x_old_na ** 2).sum())
    return (rmse / denom) < thresh if denom > F32PREC else False

def softimpute(x, lamb, maxit=1000, thresh=1e-5):
    mask = ~np.isnan(x)
    imp = x.copy()
    imp[~mask] = 0
    for _ in range(maxit):
        if x.size > 1e6:
            U, d, V = randomized_svd(imp, n_components=min(200, x.shape[1]))
        else:
            U, d, V = np.linalg.svd(imp, full_matrices=False)
        d_thresh = np.maximum(d - lamb, 0)
        rank = (d_thresh > 0).sum()
        res = U[:, :rank] @ np.diag(d_thresh[:rank]) @ V[:rank, :]
        if converged(imp, res, mask, thresh):
            break
        imp[~mask] = res[~mask]
    return U[:, :rank], imp

# --- OT Imputer ---
class OTimputer:
    def __init__(self, eps=0.01, lr=1e-2, opt=torch.optim.RMSprop, niter=2000, 
                 batchsize=128, n_pairs=1, noise=0.1, scaling=0.9):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling)

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
        X = X.clone()
        n, d = X.shape
        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape, dtype=torch.double) + self.nanmean(X, 0))[mask.bool()]
        imps.requires_grad = True
        optimizer = self.opt([imps], lr=self.lr)
        
        if X_true is not None:
            maes, rmses = np.zeros(self.niter), np.zeros(self.niter)

        for i in range(self.niter):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = sum(self.sk(X_filled[np.random.choice(n, self.batchsize)], 
                               X_filled[np.random.choice(n, self.batchsize)]) 
                      for _ in range(self.n_pairs))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if X_true is not None:
                maes[i] = self.MAE(X_filled, X_true, mask)
                rmses[i] = self.RMSE(X_filled, X_true, mask)

            if verbose and i % report_interval == 0:
                print(f"Iter {i}: Loss={loss.item()/self.n_pairs:.4f}")

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps
        return (X_filled, maes, rmses) if X_true is not None else X_filled

    @staticmethod
    def nanmean(v, dim=None):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(dim) / (~is_nan).float().sum(dim)

    @staticmethod
    def MAE(X, X_true, mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).mean()

    @staticmethod
    def RMSE(X, X_true, mask):
        mask_ = mask.bool()
        return torch.sqrt(((X[mask_] - X_true[mask_])**2).mean())
