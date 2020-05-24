import torch.optim


class LinearDecreasingLR(torch.optim.lr_scheduler.LambdaLR):
    def lr_lambda(self, iteration):
        return 1 - iteration / self._total_iters

    def __init__(self, optimizer, total_iters, last_iter=-1):
        self._total_iters = total_iters
        super().__init__(optimizer, self.lr_lambda, last_iter)


class LrSchedulerBuilder:
    def __init__(self, config):
        self.config = config['lr_scheduler']

    def build(self, optimizer, total_iters):
        if self.config['name'] == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)
        elif self.config['name'] == 'linear_decreasing':
            return LinearDecreasingLR(optimizer, total_iters)
        else:
            raise NotImplementedError(f"Unsupported LR scheduler: {self.config['name']}")
