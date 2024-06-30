import numpy as np


class SAGAOptimizer:
    def __init__(self, learning_rate=0.01, num_samples=None):
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.grad_memory = None
        self.avg_grad = None

    def update(self, params, grads, idx):
        if self.grad_memory is None:
            self.grad_memory = {key: np.zeros((self.num_samples,) + value.shape) for key, value in grads.items()}
            self.avg_grad = {key: np.zeros_like(value) for key, value in grads.items()}

        for key in params.keys():
            for i, sample_idx in enumerate(idx):
                old_grad = self.grad_memory[key][sample_idx]
                self.grad_memory[key][sample_idx] = grads[key] / len(idx)
                self.avg_grad[key] += (grads[key] / len(idx) - old_grad) / self.num_samples

            params[key] -= self.learning_rate * (grads[key] - old_grad + self.avg_grad[key])


class SVRGOptimizer:
    def __init__(self, learning_rate=0.01, num_samples=None):
        self.learning_rate = learning_rate
        self.full_grad = None

    def update(self, params, grads, snapshot_params, snapshot_grads):
        if self.full_grad is None:
            self.full_grad = {key: np.zeros_like(value) for key, value in grads.items()}

        for key in params.keys():
            self.full_grad[key] = grads[key] - snapshot_grads[key] + self.full_grad[key]

            # Update parameters
            params[key] -= self.learning_rate * self.full_grad[key]


