from imputers import OTimputer, pick_epsilon
from models import MyNet
from train_utils import train_sinkhorn_divergence
from torch.utils.tensorboard import SummaryWriter

def main():
  
    X_miss, mask, X_true = create_missing_data(X_target, missing_rate=0.3)
    
    epsilon = pick_epsilon(X_miss)
    imputer = OTimputer(eps=epsilon, batchsize=128, lr=1e-2, niter=2000)
    X_imputed, maes, rmses = imputer.fit_transform(X_miss, X_true=X_true)
    
    writer = SummaryWriter()
    model = train_sinkhorn_divergence(
        lerate=le, 
        wd=wd, 
        lra=lra,
        sources=(X_source_1, X_source_2, X_source_3, X_source_4),
        target=X_imputed,
        labels=(Y_source_1, Y_source_2, Y_source_3, Y_source_4, Y_target, Y_target_t),
        writer=writer
    )
    writer.close()

if __name__ == "__main__":
    main()
