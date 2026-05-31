# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
def test_train_MLP():
        # fixed seed for experiment
        np.random.seed(309)

        train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
        train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

        with gzip.open(train_images_path, 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

        with gzip.open(train_labels_path, 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                train_labs = np.frombuffer(f.read(), dtype=np.uint8)


        # choose 10000 samples from train set as validation set.
        idx = np.random.permutation(np.arange(num))
        # save the index.
        with open('idx.pickle', 'wb') as f:
                pickle.dump(idx, f)
        train_imgs = train_imgs[idx]
        train_labs = train_labs[idx]
        valid_imgs = train_imgs[:10000]
        valid_labs = train_labs[:10000]
        train_imgs = train_imgs[10000:]
        train_labs = train_labs[10000:]

        # normalize from [0, 255] to [0, 1]
        train_imgs = train_imgs / train_imgs.max()
        valid_imgs = valid_imgs / valid_imgs.max()

        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=100, save_dir=r'./best_models', save_name='best_model_MLP.pickle')

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

def test_train_CNN():
    np.random.seed(309)

    # load mnist
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    # shuffle
    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    # split
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    # normalize
    train_imgs = train_imgs.astype(np.float32) / 255.0
    valid_imgs = valid_imgs.astype(np.float32) / 255.0

    # CNN
    conv_configs = [
        {'in_channels': 1, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    fc_dims = [128, 10]
    cnn_model = nn.models.Model_CNN(conv_configs, fc_dims, activation='ReLU')

    # optimizer
    optimizer = nn.optimizer.SGD(init_lr=0.005, model=cnn_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500, 1500, 2500], gamma=0.5)

    # loss
    loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

    # runner
    runner = nn.runner  .RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)

    # train
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=100, save_dir=r'./best_models', save_name='best_model_CNN.pickle')

    # plot
    fig, axes = plt.subplots(1, 2)
    fig.set_tight_layout(True)
    plot(runner, axes)
    plt.show()

if __name__ == "__main__":
        test_train_MLP()
        test_train_CNN()