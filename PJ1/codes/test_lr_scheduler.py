import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import copy

def load_mnist_data():

    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    np.random.seed(309)
    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    # 归一化
    train_imgs = train_imgs.astype(np.float32) / 255.0
    valid_imgs = valid_imgs.astype(np.float32) / 255.0

    return train_imgs, train_labs, valid_imgs, valid_labs

def get_fresh_model_and_opt(lr=0.05):

    model = nn.models.Model_MLP([784, 256, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    return model, optimizer, loss_fn

def test_compare_schedulers():
    train_imgs, train_labs, valid_imgs, valid_labs = load_mnist_data()
    
    num_epochs = 20
    batch_size = 64
    base_lr = 0.05
    
    runners_dict = {}

    print("1.固定学习率")
    model1, opt1, loss1 = get_fresh_model_and_opt(base_lr)
    runner1 = nn.runner.RunnerM(model1, opt1, nn.metric.accuracy, loss1, batch_size=batch_size, scheduler=None)
    runner1.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=num_epochs, save_dir=r'./best_models', save_name='mlp_none.pickle')
    runners_dict['Constant LR'] = runner1

    print("2.阶梯衰减")
    model2, opt2, loss2 = get_fresh_model_and_opt(base_lr)
    scheduler2 = nn.lr_scheduler.MultiStepLR(optimizer=opt2, milestones=[1500, 3000], gamma=0.5)
    runner2 = nn.runner.RunnerM(model2, opt2, nn.metric.accuracy, loss2, batch_size=batch_size, scheduler=scheduler2)
    runner2.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=num_epochs, save_dir=r'./best_models', save_name='mlp_multistep.pickle')
    runners_dict['MultiStepLR'] = runner2

    print("3.指数衰减")
    model3, opt3, loss3 = get_fresh_model_and_opt(base_lr)
    # gamma=0.999 意味着每走一步迭代，学习率乘以 0.999，平滑下降
    scheduler3 = nn.lr_scheduler.ExponentialLR(optimizer=opt3, gamma=0.999)
    runner3 = nn.runner.RunnerM(model3, opt3, nn.metric.accuracy, loss3, batch_size=batch_size, scheduler=scheduler3)
    runner3.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=num_epochs, save_dir=r'./best_models', save_name='mlp_exp.pickle')
    runners_dict['ExponentialLR'] = runner3
    plot_comparison(runners_dict)


def plot_comparison(runners_dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    styles = {
        'Constant LR': {'color': '#E63946', 'linestyle': '-'},
        'MultiStepLR': {'color': '#457B9D', 'linestyle': '--'},
        'ExponentialLR': {'color': '#2A9D8F', 'linestyle': '-.'}
    }

    for name, runner in runners_dict.items():
        log_iters = 100 
        num_points = len(runner.dev_loss)
        iterations = [i * log_iters for i in range(num_points)]
        
        # 图 1：验证集 Loss 对比
        axes[0].plot(iterations, runner.dev_loss, label=name, 
                     color=styles[name]['color'], linestyle=styles[name]['linestyle'], 
                     linewidth=2, alpha=0.8)
        
        # 图 2：验证集 Accuracy 对比
        axes[1].plot(iterations, runner.dev_scores, label=name, 
                     color=styles[name]['color'], linestyle=styles[name]['linestyle'], 
                     linewidth=2, alpha=0.8)
    axes[0].set_title("Validation Loss Comparison")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')

    axes[1].set_title("Validation Accuracy Comparison")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('lr_scheduler_comparison_full.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    test_compare_schedulers()