import torch, torch.nn.functional as F
from .policy import SmallPolicy

class GRPOTrainer:
    def __init__(self, eps: float = .2, lr: float = 2e-5):
        self.policy = SmallPolicy()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr)
        self.eps = eps

    def step(self, old_logp, new_logp, adv):
        ratio = torch.exp(new_logp - old_logp)
        clip = torch.clamp(ratio, 1-self.eps, 1+self.eps)
        loss = -(torch.min(ratio*adv, clip*adv)).mean() - 1e-2 * (-(new_logp.exp()*new_logp).mean())
        self.opt.zero_grad(); loss.backward(); self.opt.step()