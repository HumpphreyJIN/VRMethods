import numpy as np
import matplotlib.pylab as plt
import os


#定义函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    x = np.where(x>=0,1,0)
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 防溢出
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #防溢出

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class Net:

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 ,W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2 ,b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = relu(a2)
        a3 = np.dot(z2,W3) + b3
        y = softmax(a3)

        return y

    def loss(self, x, t):# x:输入, t:标签
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t, lambda_ = 0.0005):# x:输入, t:标签, lanmba_ :L2正则

        l2_loss = lambda_ * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2)) / 2
        loss_W = lambda W: self.loss(x, t) + l2_loss

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])

        return grads

    def gradient(self, x, t, lambda_ = 0.0005):# x:输入, t:标签, lanmba_ :L2正则
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = relu(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        # backward
        dy = (y - t) / batch_num
        grads['W3'] = np.dot(z2.T, dy) + lambda_ * W3
        grads['b3'] = np.sum(dy, axis=0)

        da2 = np.dot(dy, W3.T)
        dz2 = relu_grad(a2) * da2
        grads['W2'] = np.dot(z1.T, dz2) + lambda_ * W2
        grads['b2'] = np.sum(dz2, axis=0)

        da1 = np.dot(da2, W2.T)
        dz1 = relu_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1) + lambda_ * W1
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def get_model(self):

        return self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'], self.params['W3'], self.params['b3']

    def set_model(self, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray):
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

    def save_parameters(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'W1.npy'), self.params['W1'])
        np.save(os.path.join(directory, 'b1.npy'), self.params['b1'])
        np.save(os.path.join(directory, 'W2.npy'), self.params['W2'])
        np.save(os.path.join(directory, 'b2.npy'), self.params['b2'])
        np.save(os.path.join(directory, 'W3.npy'), self.params['W3'])
        np.save(os.path.join(directory, 'b3.npy'), self.params['b3'])




def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

