import logging
import torch.optim


class LinearDecreasingLR(torch.optim.lr_scheduler.LambdaLR):
    def lr_lambda(self, iteration):
        return 1 - iteration / self._total_iters

    def __init__(self, optimizer, total_iters, last_iter=-1):
        self._total_iters = total_iters
        super().__init__(optimizer, self.lr_lambda, last_iter)


class WarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def lr_lambda(self, iteration):
        return self.warmup_factor if iteration <= self.warmup_iters else 1

    def __init__(self, lr_scheduler, warmup_iters, warmup_factor, last_iter=-1):
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.lr_scheduler = lr_scheduler
        super().__init__(lr_scheduler.optimizer, self.lr_lambda, lr_scheduler.last_epoch)

    def step(self, iteration=None):
        # In warmup period, call the original lr_scheduler first, then overwrite the lr with warmup LR.
        if self.last_epoch < self.warmup_iters:
            self.lr_scheduler.step(iteration)
            super().step(iteration)
        else:
            super().step(iteration)
            self.lr_scheduler.step(iteration)


class LrSchedulerBuilder:
    def __init__(self, config):
        self.config = config['lr_scheduler']

    def build(self, optimizer, total_iters):
        logging.info(f"Building a lr_scheduler. total_iters: {total_iters}, config: {self.config}")
        if self.config['name'] == 'cosine_annealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)
        elif self.config['name'] == 'linear_decreasing':
            lr_scheduler = LinearDecreasingLR(optimizer, total_iters)
        else:
            raise NotImplementedError(f"Unsupported LR scheduler: {self.config['name']}")

        warmup_iters = self.config.get('warmup_iters', 0)
        if warmup_iters > 0:
            warmup_factor = self.config.get('warmup_factor', 0.01)
            lr_scheduler = WarmupLR(lr_scheduler, warmup_iters, warmup_factor)
            logging.info(f"Using Lr Warmup: {warmup_iters} iters with {warmup_factor}")

        return lr_scheduler
