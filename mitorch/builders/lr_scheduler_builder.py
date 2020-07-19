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
        return 1  # Dummy lambda function.

    def __init__(self, lr_scheduler, warmup_iters, warmup_factor):
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.lr_scheduler = lr_scheduler
        self.true_lrs = None
        super().__init__(lr_scheduler.optimizer, self.lr_lambda, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.warmup_factor for base_lr in self.base_lrs]
        else:
            return self.true_lrs

    def step(self, iteration=None):
        # In warmup period, call the original lr_scheduler first, then overwrite the lr with warmup LR.
        if self.true_lrs:
            for group, lr in zip(self.lr_scheduler.optimizer.param_groups, self.true_lrs):
                group['lr'] = lr

        self.lr_scheduler.step()
        self.true_lrs = [group['lr'] for group in self.lr_scheduler.optimizer.param_groups]
        super().step()


class LrSchedulerBuilder:
    def __init__(self, config):
        self.config = config['lr_scheduler']
        self.max_epochs = config['max_epochs']

    def build(self, optimizer, num_epoch_iters):
        total_iters = num_epoch_iters * self.max_epochs
        logging.info(f"Building a lr_scheduler. total_iters: {total_iters}, config: {self.config}")

        if self.config['name'] == 'cosine_annealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)
        elif self.config['name'] == 'linear_decreasing':
            lr_scheduler = LinearDecreasingLR(optimizer, total_iters)
        elif self.config['name'] == 'step':
            step_size = self.config['step_size'] * num_epoch_iters
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, self.config['step_gamma'])
        else:
            raise NotImplementedError(f"Unsupported LR scheduler: {self.config['name']}")

        warmup_epochs = self.config.get('warmup_epochs', -1)
        if warmup_epochs > 0:
            warmup_iters = warmup_epochs * num_epoch_iters
            warmup_factor = self.config.get('warmup_factor', 0.01)
            lr_scheduler = WarmupLR(lr_scheduler, warmup_iters, warmup_factor)
            logging.info(f"Using Lr Warmup: {warmup_iters} iters with {warmup_factor}")

        return lr_scheduler
