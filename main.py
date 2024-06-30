import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Net import Net
from Net import _change_one_hot_label
from op import SVRGOptimizer, SAGAOptimizer


def load_data():
    x_data = np.array(np.genfromtxt('train_img.csv', dtype=int, delimiter=','))
    y_data = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')

    x_data = x_data / 255.0
    x_data = x_data.reshape(-1, 784)
    y_data = _change_one_hot_label(y_data)

    indices = np.random.permutation(len(x_data))
    train_indices = indices[:1600]
    test_indices = indices[1600:2000]

    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_test = x_data[test_indices]
    y_test = y_data[test_indices]

    return x_train, y_train, x_test, y_test


def train(network, optimizer, x_train, y_train, x_test, y_test, iters_num=200, batch_size=16):
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in tqdm(range(iters_num)):

        if isinstance(optimizer, SVRGOptimizer) and i % iter_per_epoch == 0:
            snapshot_params = {key: np.copy(value) for key, value in network.params.items()}
            snapshot_grads = network.gradient(x_train, y_train)

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grads = network.gradient(x_batch, y_batch)

        if isinstance(optimizer, (SAGAOptimizer)):
            optimizer.update(network.params, grads, batch_mask)
        elif isinstance(optimizer, SVRGOptimizer):
            optimizer.update(network.params, grads, snapshot_params, snapshot_grads)
        else:
            optimizer.update(network.params, grads)

        train_loss = network.loss(x_batch, y_batch)
        train_loss_list.append(train_loss)
        test_loss = network.loss(x_test, y_test)
        test_loss_list.append(test_loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f'Iteration {i + 1}: 训练损失 {train_loss:.4f}, 测试损失 {test_loss:.4f}, '
                  f'训练集准确率 {train_acc:.4f}, 测试集准确率 {test_acc:.4f}')

    return train_loss_list, test_loss_list, train_acc_list, test_acc_list

def plot_loss(train_loss_dict):
    for name, loss_list in train_loss_dict.items():
        plt.figure()
        plt.title(f'{name} Training Loss')
        plt.plot(loss_list, label=f'{name} Train Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def main():
    x_train, y_train, x_test, y_test = load_data()

    # 使用SAGA优化器
    optimizer_saga = SAGAOptimizer(learning_rate=0.025, num_samples=x_train.shape[0])
    network_saga = Net(input_size=784, hidden1_size=784, hidden2_size=784, output_size=10)
    train_loss_saga, _, _, _ = train(network_saga, optimizer_saga, x_train, y_train, x_test, y_test)

    # 使用SVRG优化器
    optimizer_svrg = SVRGOptimizer(learning_rate=0.005)
    network_svrg = Net(input_size=784, hidden1_size=784, hidden2_size=784, output_size=10)
    train_loss_svrg, _, _, _ = train(network_svrg, optimizer_svrg, x_train, y_train, x_test, y_test)


    # 绘制训练损失比较图
    plot_loss({
        'SAGA': train_loss_saga,
        'SVRG': train_loss_svrg
    })

if __name__ == "__main__":
    main()

