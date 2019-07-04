"""
    python3, Pytorch
    I actually don't like using **args
    but for flexiblity...
"""

from templates import Template_DecayPolicy
import numpy as np

class TimeBasedPolicy(Template_DecayPolicy):
    """
        lr *= lr_0/(1+kt)

          lr: learning rate
        lr_0: initial learning rate
           k: hyperparameter
           t: current iteration
    """

    def __init__(self, optimizer, initial_learning_rate, k, iteration_wise=True, **kwargs):
        super(TimeBasedPolicy, self).__init__(optimizer, initial_learning_rate, iteration_wise)

        self.k = k

    def decay_lr(self, curr_iter, **kwargs):
        lr = self.initial_learning_rate/(1 + self.k*curr_iter)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class StepBasedPolicy(Template_DecayPolicy):
    """
        most ordinaly laening rate scheduling

        lr = lr_0 * decay_val^floor(curr_epoch/decay_epoch)

               lr  : learning rate
               lr_0: initial learning rate
          decay_val: hyperparameter
         curr_epoch: current epoch
        decay_epoch: epoch num to decay
    """

    def __init__(self, optimizer, initial_learning_rate, decay_epoch, decay_val, iteration_wise=False, **kwargs):
        # learning rate doesn't care in this care.
        super(StepBasedPolicy, self).__init__(optimizer, initial_learning_rate, iteration_wise)

        self.decay_epoch = decay_epoch
        self.decay_val = decay_val

    def decay_lr(self, curr_epoch, **kwargs):
        lr = self.initial_learning_rate * self.decay_val**(curr_epoch//self.decay_epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class ExponentialPolicy(Template_DecayPolicy):
    """
        lr = lr_0 * exp(-kt)

          lr: learning rate
        lr_0: initial learning rate
           k: hyperparameter
           t: current iteration

    """

    def __init__(self, optimizer, initial_learning_rate, k, iteration_wise=True, **kwargs):
        super(ExponentialPolicy, self).__init__(optimizer, initial_learning_rate, iteration_wise)

        self.k = k

    def decay_lr(self, curr_iter, **kwargs):
        lr = self.initial_learning_rate * np.exp(-self.k * curr_iter)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class PolynomialPolicy(Template_DecayPolicy):
    """
        used on PSPNet

        lr = lr_0 * (1 - curr_iter/max_iter)^p

               lr: learning rate
             lr_0: initial learning rate
        curr_iter: current iteration
         max_iter: maximum iteration
                p: decaying power coefficient

    """

    def __init__(self, optimizer, initial_learning_rate, max_iter, lr_decay_power, iteration_wise=True, **kwargs):
        super(PolynomialPolicy, self).__init__(optimizer, initial_learning_rate, iteration_wise)

        self.max_iter = max_iter
        self.lr_decay_power = lr_decay_power

    def decay_lr(self, curr_iter, **kwargs):
        lr = self.initial_learning_rate * (1 - curr_iter/self.max_iter)**self.lr_decay_power

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class CosineAnnealingPolicy(Template_DecayPolicy):
    """
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, max_learning_rate, min_learning_rate, max_iter, iteration_wise=True, **kwargs):
        super(PolynomialPolicy, self).__init__(optimizer, max_learning_rate, iteration_wise)

        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_iter = max_iter

    def decay_lr(self, curr_iter, **kwargs):
        lr = self.min_learning_rate + 0.5*(self.max_learning_rate - self.min_learning_rate)*(1 + np.cos(np.pi*curr_iter/self.max_iter))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
