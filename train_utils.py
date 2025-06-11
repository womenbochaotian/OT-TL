import torch
import torch.optim as optim
import ot
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def normalize_alphas_inplace(alphas):
    alpha1, alpha2, alpha3, alpha4 = alphas
    alpha1.clamp_(min=0)
    alpha2.clamp_(min=0)
    alpha3.clamp_(min=0)
    alpha4.clamp_(min=0)
    total = alpha1 + alpha2 + alpha3 + alpha4
    alpha1.div_(total)
    alpha2.div_(total)
    alpha3.div_(total)
    alpha4.div_(total)
    return alpha1, alpha2, alpha3, alpha4

def my_loss_custom(beta, G, C0, target_predict, y):
    C = beta * C0 + torch.cdist(y, target_predict)**2
    return torch.sum(C * G)

def train_sinkhorn_divergence(lerate, wd, lra, sources, target, labels, writer=None):
    source1, source2, source3, source4 = sources
    y1, y2, y3, y4, y_target, y_target_t = labels
    target_t = target 
    
    alpha1 = alpha2 = alpha3 = alpha4 = torch.tensor(0.25, requires_grad=True)
    model = MyNet(source1.size(1))
    optimizer = optim.Adam(model.parameters(), lr=lerate, weight_decay=wd)
    optimizerA = optim.Adam([alpha1, alpha2, alpha3, alpha4], lr=lra)
    
    C1 = torch.cdist(source1, target)**2 / torch.cdist(source1, target).max()
    C2 = torch.cdist(source2, target)**2 / torch.cdist(source2, target).max()
    C3 = torch.cdist(source3, target)**2 / torch.cdist(source3, target).max()
    C4 = torch.cdist(source4, target)**2 / torch.cdist(source4, target).max()
    
    alphas_history = []
    convergence = 0
    prev_loss = float('inf')

    for k in range(1000):
        model.train()
        tp = model(target)
        a1, a2, a3, a4 = normalize_alphas_inplace([alpha1, alpha2, alpha3, alpha4])
        alphas_history.append([a1.item(), a2.item(), a3.item(), a4.item()])

        beta = a1/torch.max(C1) + a2/torch.max(C2) + a3/torch.max(C3) + a4/torch.max(C4)
        G1 = torch.from_numpy(ot.emd(np.ones(len(source1))/len(source1), 
                                     np.ones(len(target))/len(target), 
                                     (beta*C1 + torch.cdist(y1, tp)**2).detach().numpy()))
        
        loss = a1*my_loss_custom(beta, G1, C1, tp, y1) + \
               a2*my_loss_custom(beta, G2, C2, tp, y2) + \
               a3*my_loss_custom(beta, G3, C3, tp, y3) + \
               a4*my_loss_custom(beta, G4, C4, tp, y4)

        optimizer.zero_grad()
        optimizerA.zero_grad()
        loss.backward()
        optimizer.step()
        optimizerA.step()

        if abs(prev_loss - loss.item()) < 0.001:
            convergence += 1
            if convergence > 5:
                break
        else:
            convergence = 0
        prev_loss = loss.item()

        if writer and k % 10 == 0:
            writer.add_scalar('Loss', loss.item(), k)
            writer.add_scalars('Alphas', {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4}, k)

    plt.plot([h[0] for h in alphas_history], label='Source1')
    plt.plot([h[1] for h in alphas_history], label='Source2')
    plt.plot([h[2] for h in alphas_history], label='Source3')
    plt.plot([h[3] for h in alphas_history], label='Source4')
    plt.xlabel('Iterations')
    plt.ylabel('Weight')
    plt.legend()
    plt.show()
    return model
