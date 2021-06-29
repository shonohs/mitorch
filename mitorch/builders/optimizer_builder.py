import logging
import torch.optim


class OptimizerBuilder:
    def __init__(self, config):
        self.config = config.optimizer
        self.base_lr = config.lr_scheduler.base_lr

    def build(self, model):
        logging.info(f"Building a optimizer. base_lr: {self.base_lr}, config: {self.config}")
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay

        assert isinstance(self.base_lr, float) and self.base_lr > 0
        assert isinstance(momentum, float) and momentum > 0
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0  # It can be int when weight_decay == 0.

        params_no_decay = []
        params_with_decay = []
        for name, param in model.named_parameters():
            # Do not apply weight_decay to Convolution layer's Bias and Batch Norm layers.
            if name.endswith('.conv.bias') or name.endswith('.bn.weight') or name.endswith('.bn.bias'):
                params_no_decay.append(param)
            else:
                params_with_decay.append(param)

        params = [{'params': params_with_decay, 'weight_decay': weight_decay}, {'params': params_no_decay, 'weight_decay': 0}]

        if self.config.name == 'adam':
            return torch.optim.Adam(params, lr=self.base_lr, weight_decay=0)
        if self.config.name == 'sgd':
            return torch.optim.SGD(params, lr=self.base_lr, momentum=momentum, weight_decay=0)
        else:
            raise NotImplementedError(f"Non-supported optimizer: {self.config['name']}")
