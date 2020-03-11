import torch.optim


class LrSchedulerBuilder:
    def __init__(self, config):
        self.config = config['lr_scheduler']

    def build(self, optimizer, total_iters):
        if self.config['name'] == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)
        else:
            raise NotImplementedError(f"Unsupported LR scheduler: {self.config['name']}")
