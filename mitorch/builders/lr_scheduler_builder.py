import torch.optim


class LinearDecreasingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_iters, last_iter=-1):
        def lr_lambda(iteration):
            return 1 - iteration / total_iters
        super().__init__(optimizer, lr_lambda, last_iter)


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
