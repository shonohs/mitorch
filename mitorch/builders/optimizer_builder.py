import torch.optim


class OptimizerBuilder:
    def __init__(self, config):
        self.config = config['optimizer']
        self.base_lr = config['lr_scheduler']['base_lr']

    def build(self, model):
        momentum = self.config['momentum']
        weight_decay = self.config['weight_decay']

        assert isinstance(self.base_lr, float) and self.base_lr > 0
        assert isinstance(momentum, float) and momentum > 0
        assert isinstance(weight_decay, float) and weight_decay >= 0

        if self.config['name'] == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.base_lr, weight_decay=weight_decay)
        if self.config['name'] == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.base_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Non-supported optimizer: {self.config['name']}")
